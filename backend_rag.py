"""
Local File RAG Engine Backend
- Reads and indexes a local `information.txt` file
- Chunks and embeds the content at startup
- Uses Ollama nomic-embed-text for embeddings
- Uses Ollama gemma4:latest for generation

Requirements:
    pip install fastapi uvicorn httpx

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
import asyncio
import math
import os
from typing import List

app = FastAPI(title="Local File RAG Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE  = "http://localhost:11434"
EMBED_MODEL  = "nomic-embed-text:latest"
GEN_MODEL    = "gemma4:latest"
INFO_FILE    = "information.txt"

# ─────────────────────────────────────────────
#  In-memory index (populated at startup)
# ─────────────────────────────────────────────
INDEX: dict = {
    "chunks": [],       # List[str]
    "embeddings": [],   # List[List[float]]
    "loaded": False,
    "error": None,
}


# ─────────────────────────────────────────────
#  1.  CHUNKING
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
#  2.  EMBEDDINGS  (Ollama nomic-embed-text)
# ─────────────────────────────────────────────
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector for a single text. Raises on failure."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        data = r.json()
        embs = data.get("embeddings") or data.get("embedding")
        if not embs:
            raise ValueError(f"Empty embedding response from Ollama: {data}")
        return embs[0] if isinstance(embs[0], list) else embs


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts concurrently in batches to avoid overload."""
    BATCH = 10
    all_embs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        embs  = await asyncio.gather(*[get_embedding(t) for t in batch])
        all_embs.extend(embs)
        print(f"[Index] Embedded {min(i + BATCH, len(texts))}/{len(texts)} chunks...")
    return all_embs


# ─────────────────────────────────────────────
#  3.  COSINE SIMILARITY + RETRIEVAL
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
#  4.  GENERATION  (Ollama gemma4 streaming)
# ─────────────────────────────────────────────
async def generate_stream(prompt: str):
    """Stream tokens from Ollama. Yields str tokens."""
    async with httpx.AsyncClient(timeout=180) as client:
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
#  5.  STARTUP — load & index information.txt
# ─────────────────────────────────────────────
@app.on_event("startup")
async def load_and_index():
    """Read information.txt, chunk it, and embed all chunks into memory."""
    print(f"[Startup] Looking for {INFO_FILE}...")

    if not os.path.exists(INFO_FILE):
        msg = f"{INFO_FILE} not found in {os.getcwd()}"
        print(f"[Startup] ERROR: {msg}")
        INDEX["error"] = msg
        return

    with open(INFO_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        INDEX["error"] = f"{INFO_FILE} is empty."
        print(f"[Startup] ERROR: {INDEX['error']}")
        return

    print(f"[Startup] Read {len(raw_text):,} characters from {INFO_FILE}")

    chunks = chunk_text(raw_text)
    print(f"[Startup] Created {len(chunks)} chunks. Embedding now...")

    try:
        embeddings = await get_embeddings_batch(chunks)
    except Exception as e:
        INDEX["error"] = f"Embedding failed: {e}"
        print(f"[Startup] ERROR: {INDEX['error']}")
        return

    INDEX["chunks"]     = chunks
    INDEX["embeddings"] = embeddings
    INDEX["loaded"]     = True
    print(f"[Startup] ✅ Index ready — {len(chunks)} chunks embedded.")


# ─────────────────────────────────────────────
#  6.  RAG PIPELINE  (SSE endpoint)
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/rag/stream")
async def rag_stream(req: QueryRequest):
    """
    RAG pipeline over information.txt with SSE streaming.
    Event types: status | answer_token | done | error
    """

    async def event_generator():
        try:
            # Guard: index must be ready
            if INDEX["error"]:
                yield sse({"type": "error", "message": INDEX["error"]})
                return
            if not INDEX["loaded"]:
                yield sse({"type": "error", "message": "Index not ready yet. Please wait and retry."})
                return

            # ── Step 1: Embed the query ─────────────────────────────────
            yield sse({"type": "status", "message": "🧠 Embedding your question..."})
            query_emb = await get_embedding(req.query)

            # ── Step 2: Retrieve top-k chunks ───────────────────────────
            yield sse({"type": "status", "message": "🎯 Finding relevant context..."})
            top_chunks = retrieve_top_k(
                query_emb,
                INDEX["chunks"],
                INDEX["embeddings"],
                req.top_k,
            )
            context = "\n\n---\n\n".join(top_chunks)

            # ── Step 3: Build prompt ────────────────────────────────────
            prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have information about that."

CONTEXT:
{context}

USER QUESTION: {req.query}

INSTRUCTIONS:
- Answer clearly and thoroughly based on the context.
- Use markdown formatting for readability.
- Do not make up information not present in the context.

ANSWER:"""

            # ── Step 4: Stream generation ───────────────────────────────
            yield sse({"type": "status", "message": "✨ Generating answer..."})

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
#  7.  HEALTH / DEBUG ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    """Check Ollama connectivity, available models, and index status."""
    ollama_status = {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        ollama_status = {"ollama": "connected", "models": models}
    except Exception as e:
        ollama_status = {"ollama": "error", "message": str(e)}

    return {
        "status":        "ok" if INDEX["loaded"] else "not_ready",
        "index_loaded":  INDEX["loaded"],
        "index_chunks":  len(INDEX["chunks"]),
        "index_error":   INDEX["error"],
        **ollama_status,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")