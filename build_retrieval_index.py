"""
Step 4: Retrieval index — embed all brand replies for similarity search.
Usage:
    python build_retrieval_index.py --pairs data/amazon_pairs.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-npz", default="data/retrieval_index.npz")
    ap.add_argument("--out-meta", default="data/retrieval_meta.jsonl")
    args = ap.parse_args()

    pairs = []
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs", file=sys.stderr)

    texts = [p["brand_text"] for p in pairs]
    texts = [t if t.strip() else "[empty]" for t in texts]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(texts)} brand replies ...", file=sys.stderr)
    vecs = model.encode(texts, batch_size=256, show_progress_bar=True,
                        normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype=np.float32)

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz, vecs=vecs)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({
                "customer_tweet_id": p["customer_tweet_id"],
                "brand_tweet_ids": p["brand_tweet_ids"],
                "customer_text": p["customer_text"],
                "brand_text": p["brand_text"],
            }, ensure_ascii=False) + "\n")
    print(f"Index saved: {args.out_npz} ({vecs.shape})", file=sys.stderr)
    print(f"Metadata saved: {args.out_meta}", file=sys.stderr)


if __name__ == "__main__":
    main()
