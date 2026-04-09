"""
Real-time RAG Engine Backend
- Searches the web in real-time using DuckDuckGo
- Chunks and embeds retrieved content
- Uses Ollama nomic-embed-text for embeddings
- Uses Ollama gemma4:latest for generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import asyncio
import re
from typing import List, Dict, Any
import math

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
#  1.  WEB SEARCH  (DuckDuckGo HTML scrape)
# ─────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://duckduckgo.com/",
}


async def web_search(query: str, max_results: int = 8) -> List[Dict]:
    """Search DuckDuckGo HTML and return list of {title, url, snippet}."""
    results = []

    # ── Try 1: DDG HTML search (most reliable) ──────────────────────────
    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers=HEADERS,
        ) as client:
            r = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "b": "", "kl": "us-en"},
            )
            html = r.text

            # Extract result blocks
            # DDG HTML structure: <div class="result results_links...">
            result_blocks = re.findall(
                r'<div class="result[^"]*".*?</div>\s*</div>\s*</div>',
                html, re.DOTALL
            )

            for block in result_blocks[:max_results * 2]:
                # Title
                t_match = re.search(r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
                title = re.sub(r"<[^>]+>", "", t_match.group(1)).strip() if t_match else ""

                # URL – DDG wraps in redirect, extract real URL from uddg param
                u_match = re.search(r'href="([^"]+)"', block)
                raw_url = u_match.group(1) if u_match else ""
                # Decode DDG redirect
                uddg = re.search(r'uddg=([^&"]+)', raw_url)
                if uddg:
                    import urllib.parse
                    url = urllib.parse.unquote(uddg.group(1))
                else:
                    url = raw_url

                # Snippet
                s_match = re.search(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)
                snippet = re.sub(r"<[^>]+>", "", s_match.group(1)).strip() if s_match else ""

                if title and snippet:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "content": snippet,
                    })

            print(f"[Search] DDG HTML: {len(results)} results")

    except Exception as e:
        print(f"[Search] DDG HTML error: {e}")

    # ── Try 2: DDG Lite (simpler HTML, different structure) ──────────────
    if len(results) < 3:
        try:
            async with httpx.AsyncClient(
                timeout=20, follow_redirects=True, headers=HEADERS
            ) as client:
                r = await client.get(
                    "https://lite.duckduckgo.com/lite/",
                    params={"q": query, "kl": "us-en"},
                )
                html = r.text

                # Lite DDG: results in table rows
                rows = re.findall(r'<tr[^>]*>.*?</tr>', html, re.DOTALL)
                i = 0
                while i < len(rows) and len(results) < max_results:
                    row = rows[i]
                    link = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', row, re.DOTALL)
                    if link:
                        url   = link.group(1)
                        title = re.sub(r"<[^>]+>", "", link.group(2)).strip()
                        # Next row usually has snippet
                        snip_row = rows[i + 1] if i + 1 < len(rows) else ""
                        snippet  = re.sub(r"<[^>]+>", "", snip_row).strip()
                        if url.startswith("http") and title:
                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet[:300],
                                "content": snippet[:300],
                            })
                    i += 1

            print(f"[Search] DDG Lite total: {len(results)} results")

        except Exception as e:
            print(f"[Search] DDG Lite error: {e}")

    # ── Try 3: DuckDuckGo Instant Answer API ────────────────────────────
    if len(results) < 2:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                r = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                    headers=HEADERS,
                )
                data = r.json()
                if data.get("AbstractText"):
                    results.insert(0, {
                        "title": data.get("Heading", "Summary"),
                        "url":   data.get("AbstractURL", ""),
                        "snippet": data["AbstractText"],
                        "content": data["AbstractText"],
                    })
                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title":   topic["Text"][:80],
                            "url":     topic.get("FirstURL", ""),
                            "snippet": topic["Text"],
                            "content": topic["Text"],
                        })
            print(f"[Search] DDG API total: {len(results)} results")
        except Exception as e:
            print(f"[Search] DDG API error: {e}")

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for r in results:
        key = r.get("url", r.get("title", ""))[:80]
        if key and key not in seen_urls:
            seen_urls.add(key)
            unique.append(r)

    return unique[:max_results]


# ─────────────────────────────────────────────
#  2.  FETCH FULL PAGE CONTENT
# ─────────────────────────────────────────────
async def fetch_page(url: str) -> str:
    """Fetch and clean text content from a URL."""
    if not url or not url.startswith("http"):
        return ""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"},
            )
            text = r.text
            # Strip HTML tags
            text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>",  " ", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:6000]  # cap per page
    except Exception as e:
        print(f"[Fetch Error] {url}: {e}")
        return ""


# ─────────────────────────────────────────────
#  3.  CHUNKING
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────
#  4.  EMBEDDINGS  (Ollama nomic-embed-text)
# ─────────────────────────────────────────────
async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        data = r.json()
        # Ollama returns {"embeddings": [[...]]}
        embs = data.get("embeddings") or data.get("embedding")
        if isinstance(embs[0], list):
            return embs[0]
        return embs


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts concurrently."""
    tasks = [get_embedding(t) for t in texts]
    return await asyncio.gather(*tasks)


# ─────────────────────────────────────────────
#  5.  COSINE SIMILARITY + RETRIEVAL
# ─────────────────────────────────────────────
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve_top_k(
    query_emb: List[float],
    chunks: List[str],
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
    """Stream tokens from Ollama gemma4."""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": GEN_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1024,
                },
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
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
    query: str
    num_results: int = 6
    top_k: int = 5
    fetch_pages: bool = True


@app.post("/rag/stream")
async def rag_stream(req: QueryRequest):
    """
    Full RAG pipeline with SSE streaming.
    Events: status | sources | answer_token | done | error
    """

    async def event_generator():
        try:
            # ── Step 1: Web search ──────────────────────
            yield f"data: {json.dumps({'type':'status','message':'🔍 Searching the web...'})}\n\n"
            search_results = await web_search(req.query, req.num_results)

            if not search_results:
                yield f"data: {json.dumps({'type':'error','message':'No search results found.'})}\n\n"
                return

            # ── Step 2: Optionally fetch full pages ─────
            yield f"data: {json.dumps({'type':'status','message':f'📄 Fetching content from {len(search_results)} sources...'})}\n\n"

            if req.fetch_pages:
                fetch_tasks = [fetch_page(r["url"]) for r in search_results]
                pages = await asyncio.gather(*fetch_tasks)
                for i, page in enumerate(pages):
                    if page:
                        search_results[i]["content"] = page
                    # fallback to snippet
                    if not search_results[i].get("content"):
                        search_results[i]["content"] = search_results[i].get("snippet", "")

            # Emit sources
            sources = [
                {"title": r["title"], "url": r["url"], "snippet": r.get("snippet", "")[:200]}
                for r in search_results if r.get("url")
            ]
            yield f"data: {json.dumps({'type':'sources','sources':sources})}\n\n"

            # ── Step 3: Chunk all content ────────────────
            yield f"data: {json.dumps({'type':'status','message':'✂️ Chunking retrieved content...'})}\n\n"
            all_chunks = []
            for r in search_results:
                content = r.get("content", "") or r.get("snippet", "")
                if content:
                    chunks = chunk_text(content)
                    all_chunks.extend(chunks)

            if not all_chunks:
                yield f"data: {json.dumps({'type':'error','message':'Could not extract content from sources.'})}\n\n"
                return

            # Deduplicate & cap
            seen = set()
            unique_chunks = []
            for c in all_chunks:
                key = c[:100]
                if key not in seen:
                    seen.add(key)
                    unique_chunks.append(c)
            unique_chunks = unique_chunks[:60]

            # ── Step 4: Embed chunks + query ─────────────
            yield f"data: {json.dumps({'type':'status','message':f'🧠 Embedding {len(unique_chunks)} chunks...'})}\n\n"

            chunk_embs  = await get_embeddings_batch(unique_chunks)
            query_emb   = await get_embedding(req.query)

            # ── Step 5: Retrieve top-k chunks ───────────
            yield f"data: {json.dumps({'type':'status','message':'🎯 Retrieving most relevant context...'})}\n\n"
            top_chunks = retrieve_top_k(query_emb, unique_chunks, chunk_embs, req.top_k)

            context = "\n\n---\n\n".join(top_chunks)

            # ── Step 6: Build prompt ─────────────────────
            prompt = f"""You are a knowledgeable assistant with access to real-time web information.

CONTEXT FROM WEB SEARCH (most relevant excerpts):
{context}

USER QUESTION: {req.query}

INSTRUCTIONS:
- Answer the question thoroughly using the provided context.
- Cite facts from the context when possible.
- If context is insufficient, say so honestly.
- Be clear, structured, and helpful.
- Use markdown formatting for readability.

ANSWER:"""

            # ── Step 7: Stream generation ────────────────
            yield f"data: {json.dumps({'type':'status','message':'✨ Generating answer with Gemma4...'})}\n\n"

            async for token in generate_stream(prompt):
                yield f"data: {json.dumps({'type':'answer_token','token':token})}\n\n"

            yield f"data: {json.dumps({'type':'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/search/test")
async def search_test(q: str = "python programming"):
    """Debug: test web search directly."""
    results = await web_search(q, max_results=5)
    return {"query": q, "count": len(results), "results": results}


@app.get("/health")
async def health():
    """Check Ollama connectivity and available models."""
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