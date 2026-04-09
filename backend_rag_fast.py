"""
Local File RAG Engine Backend — Optimized
- Reads and indexes a local `information.txt` file
- Persistent HTTP client (no per-request connection overhead)
- Ollama /api/embed batch endpoint (all chunks in ONE request)
- NumPy vectorized cosine similarity (fast matrix dot-product)

Requirements:
    pip install fastapi uvicorn httpx numpy

Usage:
    Place `information.txt` in the same folder as this file, then run:
    python backend.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import numpy as np
import os
from typing import List

app = FastAPI(title="Local File RAG Engine — Optimized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
GEN_MODEL   = "gemma4:latest" #llama3.2:latest
INFO_FILE   = "information.txt"

# ─────────────────────────────────────────────
#  Persistent HTTP client
#  One client for the whole server lifetime.
#  Reuses TCP connections — no SSL handshake
#  overhead on every embedding/generation call.
# ─────────────────────────────────────────────
_http: httpx.AsyncClient = None  # type: ignore

# ─────────────────────────────────────────────
#  In-memory index
# ─────────────────────────────────────────────
INDEX: dict = {
    "chunks":     [],    # List[str]
    "embeddings": None,  # np.ndarray  shape (N, D)
    "loaded":     False,
    "error":      None,
}


# ─────────────────────────────────────────────
#  1.  CHUNKING
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []
    step, chunks = chunk_size - overlap, []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────
#  2.  EMBEDDINGS
#  Key optimization: send ALL texts in ONE
#  HTTP request instead of N requests.
# ─────────────────────────────────────────────
async def embed_batch(texts: List[str]) -> np.ndarray:
    """Embed multiple texts in a single Ollama API call."""
    r = await _http.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
    )
    data = r.json()
    embs = data.get("embeddings") or data.get("embedding")
    if not embs:
        raise ValueError(f"Empty embedding response: {data}")
    return np.array(embs, dtype=np.float32)


async def embed_one(text: str) -> np.ndarray:
    arr = await embed_batch([text])
    return arr[0]


# ─────────────────────────────────────────────
#  3.  RETRIEVAL  (NumPy vectorized)
#  One BLAS dot-product call over the whole
#  matrix — replaces a slow Python for-loop.
# ─────────────────────────────────────────────
def retrieve_top_k(query_emb: np.ndarray, k: int = 5) -> List[str]:
    matrix = INDEX["embeddings"]                           # (N, D)
    q      = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    scores = (matrix / norms) @ q                          # (N,)
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [INDEX["chunks"][i] for i in top_idx]


# ─────────────────────────────────────────────
#  4.  GENERATION  (streaming)
# ─────────────────────────────────────────────
async def generate_stream(prompt: str):
    async with _http.stream(
        "POST",
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model":  GEN_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.7, "num_predict": 1024},
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
#  5.  STARTUP / SHUTDOWN
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _http
    _http = httpx.AsyncClient(
        timeout=180,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )

    print(f"[Startup] Looking for {INFO_FILE}...")
    if not os.path.exists(INFO_FILE):
        INDEX["error"] = f"{INFO_FILE} not found in {os.getcwd()}"
        print(f"[Startup] ERROR: {INDEX['error']}")
        return

    with open(INFO_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        INDEX["error"] = f"{INFO_FILE} is empty."
        print(f"[Startup] ERROR: {INDEX['error']}")
        return

    print(f"[Startup] {len(raw_text):,} chars. Chunking...")
    chunks = chunk_text(raw_text)
    print(f"[Startup] {len(chunks)} chunks — embedding in ONE batch request...")

    try:
        embeddings = await embed_batch(chunks)
    except Exception as e:
        INDEX["error"] = f"Embedding failed: {e}"
        print(f"[Startup] ERROR: {INDEX['error']}")
        return

    INDEX["chunks"]     = chunks
    INDEX["embeddings"] = embeddings
    INDEX["loaded"]     = True
    print(f"[Startup] ✅ Ready — {len(chunks)} chunks, dim={embeddings.shape[1]}.")


@app.on_event("shutdown")
async def shutdown():
    if _http:
        await _http.aclose()


# ─────────────────────────────────────────────
#  6.  RAG ENDPOINT  (SSE)
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/rag/stream")
async def rag_stream(req: QueryRequest):

    async def event_generator():
        try:
            if INDEX["error"]:
                yield sse({"type": "error", "message": INDEX["error"]}); return
            if not INDEX["loaded"]:
                yield sse({"type": "error", "message": "Index not ready yet."}); return

            yield sse({"type": "status", "message": "🧠 Embedding question..."})
            query_emb = await embed_one(req.query)

            yield sse({"type": "status", "message": "🎯 Finding relevant context..."})
            top_chunks = retrieve_top_k(query_emb, req.top_k)
            context    = "\n\n---\n\n".join(top_chunks)

            prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have information about that."

CONTEXT:
{context}

QUESTION: {req.query}

INSTRUCTIONS:
- Answer clearly and thoroughly.
- Use markdown formatting.
- Do not invent information.

ANSWER:"""

            yield sse({"type": "status", "message": "✨ Generating answer..."})
            async for token in generate_stream(prompt):
                yield sse({"type": "answer_token", "token": token})

            yield sse({"type": "done"})

        except Exception as e:
            print(f"[RAG] Error: {e}")
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────
#  7.  HEALTH
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        r      = await _http.get(f"{OLLAMA_BASE}/api/tags")
        models = [m["name"] for m in r.json().get("models", [])]
        ollama = {"ollama": "connected", "models": models}
    except Exception as e:
        ollama = {"ollama": "error", "message": str(e)}

    shape = list(INDEX["embeddings"].shape) if INDEX["loaded"] else None
    return {
        "status":          "ok" if INDEX["loaded"] else "not_ready",
        "index_loaded":    INDEX["loaded"],
        "index_chunks":    len(INDEX["chunks"]),
        "embedding_shape": shape,
        "index_error":     INDEX["error"],
        **ollama,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")