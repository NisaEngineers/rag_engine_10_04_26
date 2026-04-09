# 🔮 Real-Time RAG Engine
### Gemma4 · nomic-embed-text · DuckDuckGo · FastAPI

A fully local, real-time Retrieval-Augmented Generation system that:
1. **Searches the web** live via DuckDuckGo (no API key needed)
2. **Fetches full page content** from top results
3. **Chunks** the content into overlapping passages
4. **Embeds** chunks using `nomic-embed-text` via Ollama
5. **Retrieves** the most relevant chunks via cosine similarity
6. **Generates** a streaming answer using `gemma4:latest` via Ollama

---

## 📦 Prerequisites

Make sure these Ollama models are pulled:
```bash
ollama pull gemma4:latest
ollama pull nomic-embed-text:latest
```

---

## 🚀 Setup & Run

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the RAG backend
```bash
python backend.py
```
Backend runs at: **http://localhost:8000**

Health check: http://localhost:8000/health

### 3. Open the frontend
Simply open `index.html` in your browser (double-click it).

> **Note:** The frontend makes requests to `http://localhost:8000`.  
> CORS is fully open so opening `index.html` as a local file works.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[DuckDuckGo Search] ──► top N URLs + snippets
    │
    ▼
[Page Fetcher] ──► raw HTML → clean text (async, parallel)
    │
    ▼
[Chunker] ──► overlapping 400-word chunks
    │
    ▼
[Ollama Embedder] ──► nomic-embed-text vectors (batched)
    │
    ▼
[Cosine Retriever] ──► top-K most relevant chunks
    │
    ▼
[Prompt Builder] ──► context + question
    │
    ▼
[Ollama Generator] ──► gemma4:latest streaming response
    │
    ▼
[SSE Stream] ──► frontend renders tokens live
```

---

## ⚙️ Configuration

In the UI you can adjust:
| Setting | Description | Default |
|---|---|---|
| Fetch full pages | Scrape full page text vs just snippet | ✅ On |
| Sources | Number of search results to use | 6 |
| Top-K chunks | Chunks passed to LLM as context | 5 |

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to RAG backend` | Run `python backend.py` first |
| `No search results` | Check internet connection |
| Slow embedding | Normal for CPU; nomic-embed-text is fast |
| Ollama connection refused | Make sure `ollama serve` is running |

---

## 📁 Files
```
rag_engine/
├── backend.py        ← FastAPI RAG server
├── index.html        ← Browser UI
├── requirements.txt  ← Python deps
└── README.md         ← This file
```
"# rag_engine_10_04_26" 
