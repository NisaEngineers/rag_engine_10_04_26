"""
Real-time RAG Engine Backend – Fixed Generation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import asyncio
import re
import math
import traceback
from typing import List, Dict

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

# ------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)

# ------------------------------------------------------------------
#  Web search (unchanged – works)
# ------------------------------------------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html",
}

async def web_search(query: str, max_results: int = 8) -> List[Dict]:
    results = []
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=HEADERS) as client:
            r = await client.post("https://html.duckduckgo.com/html/", data={"q": query})
            html = r.text
            blocks = re.findall(r'<div class="result[^"]*".*?</div>\s*</div>\s*</div>', html, re.DOTALL)
            for block in blocks[:max_results*2]:
                t = re.search(r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
                title = re.sub(r"<[^>]+>", "", t.group(1)).strip() if t else ""
                u = re.search(r'href="([^"]+)"', block)
                raw = u.group(1) if u else ""
                uddg = re.search(r'uddg=([^&"]+)', raw)
                url = urllib.parse.unquote(uddg.group(1)) if uddg else raw
                s = re.search(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)
                snippet = re.sub(r"<[^>]+>", "", s.group(1)).strip() if s else ""
                if title and snippet:
                    results.append({"title": title, "url": url, "snippet": snippet, "content": snippet})
    except Exception as e:
        print(f"[Search] error: {e}")
    # Deduplicate
    seen = set()
    uniq = []
    for r in results:
        key = r.get("url", r.get("title"))[:80]
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq[:max_results]

async def fetch_page(url: str) -> str:
    if not url.startswith("http"):
        return ""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "RAGBot/1.0"})
            text = re.sub(r"<script[^>]*>.*?</script>", " ", r.text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            return " ".join(text.split())[:6000]
    except Exception:
        return ""

def chunk_text(text: str, chunk_size=400, overlap=80) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{OLLAMA_BASE}/api/embed", json={"model": EMBED_MODEL, "input": text})
        data = r.json()
        embs = data.get("embeddings") or data.get("embedding")
        return embs[0] if isinstance(embs[0], list) else embs

async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    return await asyncio.gather(*[get_embedding(t) for t in texts])

def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    ma = math.sqrt(sum(x*x for x in a))
    mb = math.sqrt(sum(x*x for x in b))
    return dot/(ma*mb) if ma and mb else 0

def retrieve_top_k(q_emb, chunks, chunk_embs, k):
    scored = [(cosine_similarity(q_emb, emb), chunk) for emb, chunk in zip(chunk_embs, chunks)]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored[:k]]

# ------------------------------------------------------------------
#  CRITICAL: Robust generation with full error details
# ------------------------------------------------------------------
async def generate_stream(prompt: str):
    """
    Stream tokens from Ollama. If anything fails, yields an error token
    and prints the full traceback to the console.
    """
    print(f"[GEN] Sending prompt ({len(prompt)} chars, ~{estimate_tokens(prompt)} tokens)")
    try:
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
                        "num_ctx": 4096,
                        "stop": ["<|eot_id|>"],
                    },
                },
            ) as resp:
                print(f"[GEN] HTTP status: {resp.status_code}")
                if resp.status_code != 200:
                    body = await resp.aread()
                    error_msg = body.decode()[:200]
                    raise Exception(f"Ollama returned {resp.status_code}: {error_msg}")

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            print(f"[GEN] token: {token[:30]}...")
                            yield token
                        if data.get("done"):
                            print("[GEN] stream done")
                            break
                    except json.JSONDecodeError as e:
                        print(f"[GEN] JSON decode error: {e} – line: {line[:100]}")
                        continue
    except Exception as e:
        print(f"[GEN] EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Send a visible error to frontend
        yield f"\n\n⚠️ Generation failed: {type(e).__name__}: {e}\n"

# ------------------------------------------------------------------
#  RAG endpoint
# ------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    num_results: int = 6
    top_k: int = 5
    fetch_pages: bool = True

@app.post("/rag/stream")
async def rag_stream(req: QueryRequest):
    async def event_generator():
        try:
            yield f"data: {json.dumps({'type':'status','message':'🔍 Searching...'})}\n\n"
            results = await web_search(req.query, req.num_results)
            if not results:
                yield f"data: {json.dumps({'type':'error','message':'No results'})}\n\n"
                return

            # Fetch full pages if requested
            if req.fetch_pages:
                yield f"data: {json.dumps({'type':'status','message':'📄 Fetching pages...'})}\n\n"
                pages = await asyncio.gather(*[fetch_page(r["url"]) for r in results])
                for i, p in enumerate(pages):
                    if p:
                        results[i]["content"] = p

            # Emit sources
            sources = [{"title": r["title"], "url": r["url"], "snippet": r.get("snippet","")[:200]} for r in results]
            yield f"data: {json.dumps({'type':'sources','sources':sources})}\n\n"

            # Chunk
            yield f"data: {json.dumps({'type':'status','message':'✂️ Chunking...'})}\n\n"
            all_chunks = []
            for r in results:
                content = r.get("content", "") or r.get("snippet", "")
                if content:
                    all_chunks.extend(chunk_text(content))
            if not all_chunks:
                yield f"data: {json.dumps({'type':'error','message':'No content extracted'})}\n\n"
                return

            # Deduplicate
            seen = set()
            unique = []
            for c in all_chunks:
                key = c[:100]
                if key not in seen:
                    seen.add(key)
                    unique.append(c)
            unique = unique[:60]

            # Embed
            yield f"data: {json.dumps({'type':'status','message':f'🧠 Embedding {len(unique)} chunks...'})}\n\n"
            chunk_embs = await get_embeddings_batch(unique)
            query_emb = await get_embedding(req.query)

            # Retrieve
            yield f"data: {json.dumps({'type':'status','message':'🎯 Retrieving top chunks...'})}\n\n"
            top = retrieve_top_k(query_emb, unique, chunk_embs, req.top_k)
            context = "\n\n---\n\n".join(top)

            # Truncate context if needed (safe limit ~3000 tokens)
            MAX_TOKENS = 3000
            if estimate_tokens(context) > MAX_TOKENS:
                max_chars = int(MAX_TOKENS * 3.5)
                context = context[:max_chars]
                last = max(context.rfind('.'), context.rfind('\n'))
                if last > max_chars//2:
                    context = context[:last+1]
                print(f"[RAG] Context truncated to {len(context)} chars")

            prompt = f"""You are a helpful assistant with real-time web access.

CONTEXT:
{context}

QUESTION: {req.query}

INSTRUCTIONS: Answer using the context. Be concise and use markdown.

ANSWER:"""

            yield f"data: {json.dumps({'type':'status','message':'✨ Generating answer...'})}\n\n"
            token_count = 0
            async for token in generate_stream(prompt):
                if token:
                    token_count += 1
                    yield f"data: {json.dumps({'type':'answer_token','token':token})}\n\n"
                await asyncio.sleep(0)

            if token_count == 0:
                yield f"data: {json.dumps({'type':'answer_token','token':'⚠️ No response from model. Check backend logs.'})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"

        except Exception as e:
            print(f"[RAG] Unhandled: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ------------------------------------------------------------------
#  Test endpoints
# ------------------------------------------------------------------
@app.get("/test/generate")
async def test_generate(q: str = "Say hello"):
    full = ""
    async for token in generate_stream(f"Answer concisely: {q}"):
        full += token
    return {"response": full, "length": len(full)}

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {"status": "ok", "models": models}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")