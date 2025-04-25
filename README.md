# Retrieval-Augmented-Generation (RAG) Micro-Stack

A plug-and-play toolkit that turns a folder of **plain-text documents** into a searchable knowledge base fronted by a light-weight API.  
Under the hood:

| Layer              | Technology                | Purpose                                    |
|--------------------|---------------------------|--------------------------------------------|
| **Vector DB**      | **Milvus v2**             | Stores embeddings & performs ANN search    |
| **Embedding**      | *Sentence-Transformers*   | Converts text chunks → dense vectors       |
| **LLM**            | **Ollama** runtime        | Generates answers from context             |
| **API**            | **FastAPI**               | `POST /chat` endpoint for client apps      |
| **Orchestration**  | Docker Compose            | One-command bootstrap                      |

---

## 1 - Project Layout

```
.
├── app.py                          # FastAPI service
├── rag.py                          # Prompt builder + retrieval utilities
├── vectorstore.py                  # One-shot ingestion script
├── utils.py                        # Chunking helpers
├── config.py                       # Centralised settings (env-driven)
├── data/                               # 📂 Drop .txt files here
│   └── *.txt
├── Dockerfile                      # Base image for Python services
├── Dockerfile.ollama               # Slim wrapper that `ollama pull`s your model
├── docker-compose.yml              # Milvus + Ollama + API + ingest job
├── docker-compose.override.yml     # Optional automatic reload FastAPI on code changes
└── pyproject.toml                  # Poetry dependencies
```

> The `init-db` service uses a SHA-1 checksum to auto-re-ingest when `data/` changes—no manual rebuilds.

---

## 2 - Quick Start (Docker)

1. Clone the repo and add your `.txt` files:
   ```bash
   git clone <repo-url> && cd <repo>
   mkdir -p data && cp ~/your-docs/*.txt data/
   ```
2. Launch the stack:
   ```bash
   docker compose up -d
   ```
3. Verify health:
   - Milvus & collection readiness
   - Ollama model loaded
   - API healthy at `http://localhost:8000/healthz`

---

## 3 - Using the API

Send a chat request:

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question":"What problems does NAT solve?"}'
```

Sample response:

```jsonc
{
  "prompt": "...full prompt sent to the LLM...",
  "answer": "Network Address Translation (NAT) allows..."
}
```

---

## 4 - Configuration

All settings live in `config.py` and respect environment variables:

| Env Var           | Default                                | Description                                 |
|-------------------|----------------------------------------|---------------------------------------------|
| `MILVUS_HOST`     | `localhost`                            | Vector-DB host                              |
| `MILVUS_PORT`     | `19530`                                | Vector-DB port                              |
| `COLLECTION`      | `main`                                 | Milvus collection name                      |
| `EMBED_MODEL`     | `nomic-ai/nomic-embed-text-v1.5`       | Sentence-Transformer model                  |
| `CHUNK_SIZE`      | `128`                                  | Words per chunk                             |
| `OVERLAP`         | `64`                                   | Words overlap between chunks                |
| `TOP_K`           | `5`                                    | Number of hits to retrieve                  |
| `NPROBE`          | `10`                                   | Milvus search recall parameter              |
| `LLM_HOST`        | `localhost`                            | Ollama host                                 |
| `LLM_PORT`        | `11434`                                | Ollama port                                 |
| `LLM_MODEL`       | `llama3.1:8b`                          | Ollama-served LLM model                     |
| `LLM_TIMEOUT`     | `180`                                  | Seconds before LLM client times out         |
| `LLM_TEMPERATURE` | `0.7`                                  | Sampling randomness                         |
| `LLM_MAX_TOKENS`  | `256`                                  | Max response tokens                         |

**Pro Tip:** Use a single `.env` file or Compose override to inject these vars; `config.py` reads them automatically.

---

## 5 - Re-Ingesting or Updating Data

- **Via Docker:**  
  Just add/edit `.txt` in `data/` and `docker compose up -d` again.  
  The `init-db` container detects changes, rebuilds vectors, and recreates the collection.
- **Locally (no Docker):**  
  ```bash
  poetry install
  python vectorstore.py --data-dir data --collection my_collection
  ```

---

## 6 - Customisation & Extension

| Task                                    | File / Location                 | How to Tweak                                            |
|-----------------------------------------|---------------------------------|---------------------------------------------------------|
| Change prompt template                  | `rag.py ➔ build_prompt()`      | Edit the multi-line f-string                            |
| Swap embedding model                    | Env var                         | `EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"`  |
| Use a different LLM                     | Env var                         | `LLM_MODEL="mistral:7b-instruct"`                       |
| Support PDF/HTML ingestion              | `utils.py` + `vectorstore.py`   | Plug in a parser, feed text into `chunk_text()`         |
| Scale Milvus (cluster mode)             | `docker-compose.yml`            | Switch to a Milvus cluster image & topology             |
| Migrate to another vector DB            | `vectorstore.py` + `rag.py`     | Abstract the client calls behind an adapter             |

---

## 7 - Troubleshooting

| Symptom                                      | Likely Cause                              | Remedy                                           |
|----------------------------------------------|-------------------------------------------|--------------------------------------------------|
| `LLM error: EOF`                             | Model still loading / OOM                 | Increase `LLM_TIMEOUT` or choose smaller model   |
| `RuntimeError: Milvus collection 'main'...`  | Ingestion failed or skipped               | Inspect `init-db` logs; confirm `.txt` presence  |
| Slow ingestion / high memory usage           | Too large `CHUNK_SIZE` or heavy model     | Decrease chunk size or select lighter embedder   |

---

## 8 - Roadmap Ideas

- **Hot-reload** watcher to auto-ingest new docs in real time  
- **Streaming** responses from the LLM for lower latency  
- **Auth & rate-limiting** middleware for production-grade API  

> Contributions welcome! Let’s co-innovate and supercharge your domain knowledge. 🚀
