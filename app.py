#!/usr/bin/env python3
"""
app.py – FastAPI RAG service, now wired to Ollama-Serve (HTTP) & persistent Milvus connection.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymilvus import connections, Collection
import httpx

from rag import embed, build_prompt
from config import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION,
    LLM_HOST, LLM_PORT, LLM_MODEL,
    LLM_TIMEOUT, NPROBE, TOP_K,
)

LLM_URL = f"http://{LLM_HOST}:{LLM_PORT}/api/generate"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RaveCraft RAG")

# ─── Connect to Milvus and initialize Collection on startup ───────────────────
@app.on_event("startup")
async def startup_event():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    app.state.collection = Collection(COLLECTION)
    app.state.collection.load()
    logger.info(f"Connected to Milvus collection '{COLLECTION}'")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "collection"):
        app.state.collection.release()
    connections.disconnect("default")

# ─── Schemas ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    prompt: str
    answer: str

async def call_llm(prompt: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        resp = await client.post(LLM_URL, json=payload)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=resp.text)
    return resp.json().get("response", "").strip()

# Define retrieve here instead of importing
def retrieve(collection: Collection, vec: list, k: int = TOP_K, nprobe: int = NPROBE) -> list:
    hits = collection.search(
        [vec], "embedding",
        param={"metric_type": "IP", "params": {"nprobe": nprobe}},
        limit=k,
        output_fields=["chunk"]
    )
    return [h.entity.get("chunk") for h in hits[0]]

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info(f"Received question: {req.question}")
    vec = embed(req.question)
    chunks = retrieve(app.state.collection, vec)
    prompt = build_prompt(chunks, req.question)

    try:
        logger.info("Calling LLM...")
        answer = await call_llm(prompt)
        logger.info("LLM response received")
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return JSONResponse(
            status_code=502,
            content={"prompt": prompt, "answer": f"LLM error: {e}"}
        )
    return ChatResponse(prompt=prompt, answer=answer)
