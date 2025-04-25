# config.py – single source of truth for env-driven settings
import os

# Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION  = os.getenv("COLLECTION",  "ravecraft")

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE", "128"))
OVERLAP     = int(os.getenv("OVERLAP",    "64"))

# Retrieval
TOP_K   = int(os.getenv("TOP_K",   "5"))
NPROBE  = int(os.getenv("NPROBE",  "10"))

# LLM (Ollama)
LLM_HOST        = os.getenv("LLM_HOST", "localhost")
LLM_PORT        = int(os.getenv("LLM_PORT", "11434"))
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_TIMEOUT     = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "256"))
