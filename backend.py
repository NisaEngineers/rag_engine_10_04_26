"""
Real-time RAG Engine Backend
- Searches the web in real-time using duckduckgo-search library
- Chunks and embeds retrieved content
- Uses Ollama nomic-embed-text for embeddings
- Uses Ollama gemma4:latest for generation

Requirements:
    pip install fastapi uvicorn httpx ddgs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import asyncio
import re
from typing import List, Dict
import math
from ddgs import DDGS

app = FastAPI(title="Real-Time RAG Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
GEN_MODEL   = "gemma4:latest"


# ─────────────────────────────────────────────
#  1.  WEB SEARCH  (duckduckgo-search library)
# ─────────────────────────────────────────────
async def web_search(query: str, max_results: int = 8) -> List[Dict]:
    """
    Search using the duckduckgo-search library.
    Runs in a thread pool to avoid blocking the event loop.
    Returns list of {title, url, snippet, content}.
    """
    def _search():
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title":   r.get("title", ""),
                        "url":     r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "content": r.get("body", ""),
                    })
        except Exception as e:
            print(f"[Search] DDGS error: {e}")
        return results

    results = await asyncio.get_event_loop().run_in_executor(None, _search)
    print(f"[Search] Got {len(results)} results for: {query!r}")
    return results


# ─────────────────────────────────────────────
#  2.  FETCH FULL PAGE CONTENT
# ─────────────────────────────────────────────
async def fetch_page(url: str) -> str:
    """Fetch and clean text content from a URL. Returns '' on any failure."""
    if not url or not url.startswith("http"):
        return ""
    try:
        async with httpx.AsyncClient(
            timeout=10,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"},
        ) as client:
            r = await client.get(url)
            if r.status_code != 200:
                print(f"[Fetch] {url} returned HTTP {r.status_code}")
                return ""
            text = r.text
            text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>",   " ", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:6000]
    except Exception as e:
        print(f"[Fetch] Error fetching {url}: {e}")
        return ""


# ─────────────────────────────────────────────
#  3.  CHUNKING
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────
#  4.  EMBEDDINGS  (Ollama nomic-embed-text)
# ─────────────────────────────────────────────
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector for a single text. Raises on failure."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        data = r.json()
        embs = data.get("embeddings") or data.get("embedding")
        if not embs:
            raise ValueError(f"Empty embedding response from Ollama: {data}")
        # Ollama /api/embed returns {"embeddings": [[float, ...]]}
        return embs[0] if isinstance(embs[0], list) else embs


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts concurrently."""
    tasks = [get_embedding(t) for t in texts]
    return await asyncio.gather(*tasks)


# ─────────────────────────────────────────────
#  5.  COSINE SIMILARITY + RETRIEVAL
# ─────────────────────────────────────────────
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve_top_k(
    query_emb:  List[float],
    chunks:     List[str],
    chunk_embs: List[List[float]],
    k: int = 5,
) -> List[str]:
    scored = [
        (cosine_similarity(query_emb, emb), chunk)
        for emb, chunk in zip(chunk_embs, chunks)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]


# ─────────────────────────────────────────────
#  6.  GENERATION  (Ollama gemma4 streaming)
# ─────────────────────────────────────────────
async def generate_stream(prompt: str):
    """Stream tokens from Ollama. Yields str tokens."""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model":  GEN_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1024,
                },
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data  = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


# ─────────────────────────────────────────────
#  7.  RAG PIPELINE  (SSE endpoint)
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:       str
    num_results: int  = 6
    top_k:       int  = 5
    fetch_pages: bool = True


def sse(payload: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/rag/stream")
async def rag_stream(req: QueryRequest):
    """
    Full RAG pipeline with SSE streaming.
    Event types: status | sources | answer_token | done | error
    """

    async def event_generator():
        try:
            # ── Step 1: Web search ──────────────────────────────────────
            yield sse({"type": "status", "message": "🔍 Searching the web..."})
            search_results = await web_search(req.query, req.num_results)

            if not search_results:
                yield sse({"type": "error", "message": "No search results found. Try a different query."})
                return

            # ── Step 2: Fetch full pages (optional) ────────────────────
            yield sse({"type": "status", "message": f"📄 Fetching content from {len(search_results)} sources..."})

            if req.fetch_pages:
                pages = await asyncio.gather(*[fetch_page(r["url"]) for r in search_results])
                for i, page_text in enumerate(pages):
                    if page_text:
                        search_results[i]["content"] = page_text
                    elif not search_results[i].get("content"):
                        # Final fallback: use snippet
                        search_results[i]["content"] = search_results[i].get("snippet", "")

            # Emit sources to the client
            sources = [
                {
                    "title":   r["title"],
                    "url":     r["url"],
                    "snippet": r.get("snippet", "")[:200],
                }
                for r in search_results if r.get("url")
            ]
            yield sse({"type": "sources", "sources": sources})

            # ── Step 3: Chunk content ───────────────────────────────────
            yield sse({"type": "status", "message": "✂️ Chunking retrieved content..."})

            all_chunks: List[str] = []
            for r in search_results:
                content = r.get("content") or r.get("snippet", "")
                if content:
                    all_chunks.extend(chunk_text(content))

            print(f"[RAG] Total raw chunks: {len(all_chunks)}")

            if not all_chunks:
                yield sse({"type": "error", "message": "Could not extract any content from sources."})
                return

            # Deduplicate & cap
            seen: set = set()
            unique_chunks: List[str] = []
            for c in all_chunks:
                key = c[:100]
                if key not in seen:
                    seen.add(key)
                    unique_chunks.append(c)
            unique_chunks = unique_chunks[:60]
            print(f"[RAG] Unique chunks after dedup: {len(unique_chunks)}")

            # ── Step 4: Embed ───────────────────────────────────────────
            yield sse({"type": "status", "message": f"🧠 Embedding {len(unique_chunks)} chunks..."})

            chunk_embs = await get_embeddings_batch(unique_chunks)
            query_emb  = await get_embedding(req.query)

            # ── Step 5: Retrieve top-k ──────────────────────────────────
            yield sse({"type": "status", "message": "🎯 Retrieving most relevant context..."})
            top_chunks = retrieve_top_k(query_emb, unique_chunks, chunk_embs, req.top_k)
            context    = "\n\n---\n\n".join(top_chunks)

            # ── Step 6: Build prompt ────────────────────────────────────
            prompt = f"""You are a knowledgeable assistant with access to real-time web information.

CONTEXT FROM WEB SEARCH (most relevant excerpts):
{context}

USER QUESTION: {req.query}

INSTRUCTIONS:
- Answer the question thoroughly using the provided context.
- Cite facts from the context when possible.
- If the context is insufficient, say so honestly.
- Be clear, structured, and helpful.
- Use markdown formatting for readability.

ANSWER:"""

            # ── Step 7: Stream generation ───────────────────────────────
            yield sse({"type": "status", "message": "✨ Generating answer with Gemma4..."})

            async for token in generate_stream(prompt):
                yield sse({"type": "answer_token", "token": token})

            yield sse({"type": "done"})

        except Exception as e:
            print(f"[RAG] Unhandled error: {e}")
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────
#  8.  DEBUG / HEALTH ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/search/test")
async def search_test(q: str = "python programming"):
    """Debug: run a web search and return raw results."""
    results = await web_search(q, max_results=5)
    return {"query": q, "count": len(results), "results": results}


@app.get("/health")
async def health():
    """Check Ollama connectivity and list available models."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {"status": "ok", "ollama": "connected", "models": models}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
