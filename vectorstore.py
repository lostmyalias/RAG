#!/usr/bin/env python3
"""
vectorstore.py — Unified vectorization & Milvus ingestion
"""
import sys, argparse, pathlib, textwrap
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType, Collection
)
from config import (
    EMBED_MODEL,
    MILVUS_HOST, MILVUS_PORT, COLLECTION as COLLECTION_NAME,
)
from utils import chunk_text, get_txt_files

def process_documents(model_name: str, files: List[pathlib.Path]) -> List[dict]:
    """Embed each chunk with the nominated model."""
    model = SentenceTransformer(model_name, trust_remote_code=True)
    records = []
    for fp in tqdm(files, desc="Processing files"):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunk_text(text):
            vec = model.encode(chunk, normalize_embeddings=True)
            records.append({
                "source": str(fp),
                "chunk": chunk,
                "vector": vec.tolist()
            })
    return records

def create_collection(name: str, dim: int) -> Collection:
    if utility.has_collection(name):
        utility.drop_collection(name)
    fields = [
        FieldSchema("id",        dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema("source",    dtype=DataType.VARCHAR,       max_length=512),
        FieldSchema("chunk",     dtype=DataType.VARCHAR,       max_length=4096),
        FieldSchema("embedding", dtype=DataType.FLOAT_VECTOR,  dim=dim),
    ]
    schema = CollectionSchema(fields, description=name)
    return Collection(name, schema, shards_num=1)

def main():
    p = argparse.ArgumentParser(description=textwrap.dedent("Process ./data → Milvus"))
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--model",      default=EMBED_MODEL)
    p.add_argument("--collection", default=COLLECTION_NAME)
    p.add_argument("--host",       default=MILVUS_HOST)
    p.add_argument("--port",       default=MILVUS_PORT)
    args = p.parse_args()

    data_path = pathlib.Path(args.data_dir)
    files = get_txt_files(data_path)
    if not files:
        sys.exit(f"No .txt files found in {data_path}")

    print(f"[1/5] Embedding {len(files)} docs with {args.model}…")
    records = process_documents(args.model, files)

    print(f"[2/5] Connecting to Milvus @ {args.host}:{args.port}…")
    connections.connect(host=args.host, port=args.port)

    dim = len(records[0]["vector"])
    print(f"[3/5] Creating collection '{args.collection}' (dim={dim})…")
    coll = create_collection(args.collection, dim)

    print(f"[4/5] Inserting {len(records):,} vectors…")
    sources = [r["source"] for r in records]
    chunks  = [r["chunk"]  for r in records]
    vecs    = [r["vector"] for r in records]
    res = coll.insert([sources, chunks, vecs])

    print("[5/5] Building index + loading…")
    coll.create_index(field_name="embedding",
                      index_params={"index_type":"IVF_FLAT","params":{"nlist":128},"metric_type":"IP"})
    coll.load()
    print(f"✓ Done: {len(res.primary_keys):,} vectors in '{args.collection}'")

if __name__ == "__main__":
    main()



