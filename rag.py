"""
rag.py — core Retrieval-Augmented Generation utilities + CLI helpers.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from config import (
    EMBED_MODEL, MILVUS_HOST, MILVUS_PORT,
    COLLECTION, TOP_K, NPROBE,
)
from utils import chunk_text, get_txt_files

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Embedder ────────────────────────────────────────────────────────────────
_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)

def embed(text: str) -> list[float]:
    """Return a normalized embedding for *text*."""
    return _model.encode(text, normalize_embeddings=True).tolist()

# ─── Vector search ───────────────────────────────────────────────────────────
def retrieve(vec: list[float],
             collection: Collection | None = None,
             host: str = MILVUS_HOST,
             port: int = MILVUS_PORT,
             coll_name: str = COLLECTION,
             k: int = TOP_K,
             nprobe: int = NPROBE) -> list[str]:

    if collection is None:
        if not connections.has_connection("default"):
            connections.connect(host=host, port=port)
        collection = Collection(coll_name)

    hits = collection.search(
        [vec], "embedding",
        param={"metric_type": "IP", "params": {"nprobe": nprobe}},
        limit=k,
        output_fields=["chunk"],
    )
    chunks = [h.entity.get("chunk") for h in hits[0]]
    logger.info("retrieve(): %d chunks (k=%d, nprobe=%d)", len(chunks), k, nprobe)
    return chunks

# ─── Prompt builder ──────────────────────────────────────────────────────────
def build_prompt(chunks: list[str], question: str) -> str:
    ctx = "\n\n---\n\n".join(chunks)
    return textwrap.dedent(f"""
        You are **RaveCraft-GPT**, the domain-expert assistant for the Shopify store
        https://ea4mn7-jq.myshopify.com.

        **Ground rules**
        • Base every reply solely on the <Context> block.  
        • If the context lacks the answer, say  
          “I’m sorry, I don’t have that information right now.”  
        • When describing navigation, use clear paths  
          (e.g., “Home → Collections → CyberPulse 3D LED Glasses”).  
        • Quote product titles, prices, or button labels exactly as written.  
        • Do **not** invent URLs or details.  
        • Default to ≤ 4 concise sentences unless depth is requested.

        <Context>
        {ctx}
        </Context>

        <Question>
        {question}
        </Question>

        <Answer>
    """).strip()

# ─── CLI entry-points ────────────────────────────────────────────────────────
def _cli_embed(args):
    print(embed(args.question))

def _cli_retrieve(args):
    chunks = retrieve(
        args.vector,
        host=args.host,
        port=args.port,
        coll_name=args.collection,
        k=args.k,
        nprobe=args.nprobe,
    )
    print("\n\n".join(chunks))

def _cli_prompt(args):
    print(build_prompt(args.chunks, args.question))

def cli() -> None:
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="Encode a question")
    e.add_argument("question")
    e.set_defaults(func=_cli_embed)

    r = sub.add_parser("retrieve", help="Vector search")
    r.add_argument("vector", nargs="+", type=float)
    r.add_argument("--host", default=MILVUS_HOST)
    r.add_argument("--port", default=MILVUS_PORT)
    r.add_argument("--collection", default=COLLECTION)
    r.add_argument("-k", type=int, default=TOP_K)
    r.add_argument("--nprobe", type=int, default=NPROBE)
    r.set_defaults(func=_cli_retrieve)

    pp = sub.add_parser("prompt", help="Craft prompt")
    pp.add_argument("question")
    pp.add_argument("chunks", nargs="+")
    pp.set_defaults(func=_cli_prompt)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    cli()
