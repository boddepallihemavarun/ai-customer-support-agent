"""
Step 5: Retrieval demo — type a customer tweet, get similar past cases
with Amazon's actual replies.
Usage:
    python search_demo.py --query "my package was supposed to arrive yesterday"
    python search_demo.py            (interactive mode)
"""

import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=None)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--index", default="data/retrieval_index.npz")
    ap.add_argument("--meta", default="data/retrieval_meta.jsonl")
    args = ap.parse_args()

    print("Loading index ...", file=sys.stderr)
    vecs = np.load(args.index)["vecs"]
    meta = []
    with open(args.meta, encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    print(f"Index ready: {vecs.shape[0]} past cases", file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(q):
        qv = model.encode([q], normalize_embeddings=True)[0]
        sims = vecs @ qv
        top = np.argsort(-sims)[:args.top]
        print("\n" + "=" * 80)
        print(f"QUERY: {q}\n")
        for rank, i in enumerate(top, 1):
            m = meta[i]
            print(f"--- Match {rank}  (similarity {sims[i]:.3f}) ---")
            print(f"  Past customer : {m['customer_text'][:160]}")
            print(f"  Amazon replied: {m['brand_text'][:160]}")
            print(f"  tweet ids     : customer {m['customer_tweet_id']}, "
                  f"brand {m['brand_tweet_ids']}")
        print("=" * 80)

    if args.query:
        search(args.query)
        return

    print("\nInteractive mode — type a customer tweet (or 'quit'):")
    while True:
        q = input("\nquery> ").strip()
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        search(q)


if __name__ == "__main__":
    main()
