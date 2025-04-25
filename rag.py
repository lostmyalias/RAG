# rag.py — core RAG utils + CLI
import argparse
import textwrap
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from config import (
    EMBED_MODEL,MILVUS_HOST, MILVUS_PORT, COLLECTION, TOP_K, NPROBE,
)
from utils import chunk_text, get_txt_files

# Initialize model once at module level
_model = SentenceTransformer(EMBED_MODEL)

def embed(question, model_name=EMBED_MODEL):
    if model_name != EMBED_MODEL:
        raise ValueError(f"Model switching not supported. Using {EMBED_MODEL}")
    return _model.encode(question, normalize_embeddings=True).tolist()

def retrieve(vec,
            collection=None,
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            coll_name=COLLECTION,
            k=TOP_K,
            nprobe=NPROBE):
    """Search for similar chunks using vector embedding."""
    if collection is None:
        if not connections.has_connection("default"):
            connections.connect(host=host, port=port)
        collection = Collection(coll_name)
        
    hits = collection.search(
        [vec], "embedding",
        param={"metric_type": "IP", "params": {"nprobe": nprobe}},
        limit=k,
        output_fields=["chunk"]
    )
    return [h.entity.get("chunk") for h in hits[0]]

def build_prompt(chunks, question):
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
        • If the question is ambiguous, ask one clarifying question instead of guessing.  
        • Default to ≤ 4 concise sentences unless the user explicitly asks for depth.

        <Context>
        {ctx}
        </Context>

        <Question>
        {question}
        </Question>

        <Answer>
    """).strip()



# ── CLI entry-points ─────────────────────────────────────────────────────────

def _cli_embed(args):
    vec = embed(args.question, args.model)
    print(vec)

def _cli_retrieve(args):
    chunks = retrieve(args.vector, args.host, args.port, args.collection, args.k, args.nprobe)
    for c in chunks:
        print(c)

def _cli_prompt(args):
    prompt = build_prompt(args.chunks, args.question)
    print(prompt)

def cli():
    p = argparse.ArgumentParser(prog="rag")
    subs = p.add_subparsers(dest="cmd", required=True)

    e = subs.add_parser("embed", help="Encode a question")
    e.add_argument("question", help="User prompt text")
    e.add_argument("--model", default=EMBED_MODEL)
    e.set_defaults(func=_cli_embed)

    r = subs.add_parser("retrieve", help="Retrieve top-k chunks")
    r.add_argument("vector", nargs="+", type=float, help="Embedding vector")
    r.add_argument("--host",       default=MILVUS_HOST)
    r.add_argument("--port",       default=MILVUS_PORT)
    r.add_argument("--collection", default=COLLECTION)
    r.add_argument("-k", type=int, default=TOP_K)
    r.add_argument("--nprobe", type=int, default=NPROBE)
    r.set_defaults(func=_cli_retrieve)

    p_ = subs.add_parser("prompt", help="Build the LLM prompt")
    p_.add_argument("question", help="User question")
    p_.add_argument("chunks", nargs="+", help="Retrieved context chunks")
    p_.set_defaults(func=_cli_prompt)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    cli()
