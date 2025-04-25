#!/usr/bin/env python3
"""
app.py – FastAPI RAG service wired to Ollama & Milvus.
Adds (a) retry/back-off for Milvus, (b) LLM pre-warm.
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymilvus import connections, Collection, utility
import httpx

from rag     import embed, build_prompt
from config  import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION,
    LLM_HOST,    LLM_PORT,   LLM_MODEL,
    LLM_TIMEOUT, TOP_K, NPROBE,
)

LLM_URL = f"http://{LLM_HOST}:{LLM_PORT}/api/generate"

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RaveCraft RAG")

# ── Milvus bootstrap + LLM pre-warm ──────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("🔗 Connecting to Milvus @ %s:%s", MILVUS_HOST, MILVUS_PORT)
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    for attempt in range(1, 6):                    # 5 retries, 5 s apart
        if utility.has_collection(COLLECTION):
            app.state.collection = Collection(COLLECTION)
            app.state.collection.load()
            logger.info("✅ Loaded collection '%s'", COLLECTION)
            break
        if attempt == 5:
            raise RuntimeError(f"Milvus collection '{COLLECTION}' missing")
        logger.warning("⏳ Collection not ready (try %d/5)…", attempt)
        await asyncio.sleep(5)

    # ── Pre-warm LLM so first user call isn’t slow ───────────────────────────
    try:
        logger.info("⚡ Pre-warming LLM runner…")
        await call_llm("ping")          #  trivial prompt
        logger.info("✅ LLM ready")
    except Exception as exc:            # noqa: BLE001
        logger.warning("LLM pre-warm failed: %s (continuing)", exc)

@app.on_event("shutdown")
async def shutdown_event() -> None:
    if hasattr(app.state, "collection"):
        app.state.collection.release()
    connections.disconnect("default")

# ── Schemas ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    prompt: str
    answer: str

# ── Internal helpers ─────────────────────────────────────────────────────────
async def call_llm(prompt: str) -> str:
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        resp = await client.post(LLM_URL, json=payload)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=resp.text)
    return resp.json().get("response", "").strip()

def retrieve(collection: Collection,
             vec: list[float],
             k: int = TOP_K,
             nprobe: int = NPROBE) -> list[str]:
    hits = collection.search(
        [vec], "embedding",
        param={"metric_type": "IP", "params": {"nprobe": nprobe}},
        limit=k,
        output_fields=["chunk"],
    )
    return [h.entity.get("chunk") for h in hits[0]]

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info("❓ %s", req.question)
    vec    = embed(req.question)
    chunks = retrieve(app.state.collection, vec)
    prompt = build_prompt(chunks, req.question)

    try:
        answer = await call_llm(prompt)
    except Exception as exc:            # noqa: BLE001
        logger.exception("LLM error")
        return JSONResponse(
            status_code=502,
            content={"prompt": prompt, "answer": f"LLM error: {exc}"},
        )
    return ChatResponse(prompt=prompt, answer=answer)
