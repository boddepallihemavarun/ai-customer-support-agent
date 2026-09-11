"""
Taxonomy Derivation — AmazonHelp Intents
=========================================
Step 3: Derive the intent taxonomy FROM THE DATA.

    1. Sample customer messages from the cleaned pairs (Step 2 output).
    2. Embed them (sentence-transformers) and cluster (KMeans, k = 4..16).
    3. Print representative messages + keywords per cluster so YOU can
       name each intent (human-in-the-loop).
    4. Emit a draft taxonomy JSON.
    5. Re-run with --labels to lock names; --full-label assigns intents
       to the FULL corpus for the feasibility check.

Usage (PowerShell, ONE line each):
    python derive_taxonomy.py --pairs data/amazon_pairs.jsonl
    python derive_taxonomy.py --pairs data/amazon_pairs.jsonl --labels results/my_labels.json --full-label

Dependencies:
    pip install sentence-transformers scikit-learn numpy
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def sample_messages(pairs, n, seed=42, min_len=15, max_len=280):
    rng = random.Random(seed)
    pool = []
    for p in pairs:
        t = p.get("customer_text", "").strip()
        if not (min_len <= len(t) <= max_len):
            continue
        low = t.lower()
        if re.fullmatch(r"(thank(s| you)[!. ]*)+|ok(ay)?[!. ]*|yes[!. ]*|no[!. ]*", low):
            continue
        pool.append(t)
    rng.shuffle(pool)
    return pool[:n]


def embed(texts, model_name="all-MiniLM-L6-v2", batch_size=256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(texts)} messages with {model_name} ...", file=sys.stderr)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)


def choose_k(vecs, k_min, k_max, seed=42):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    results = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(vecs)
        score = silhouette_score(vecs, km.labels_,
                                 sample_size=min(5000, len(vecs)), random_state=seed)
        results[k] = (score, km)
        print(f"  k={k:>2}  silhouette={score:.4f}", file=sys.stderr)
    best_k = max(results, key=lambda k: results[k][0])
    print(f"Best k = {best_k} (silhouette={results[best_k][0]:.4f})", file=sys.stderr)
    return best_k, results[best_k][1]


def cluster_keywords(texts, labels, top_n=12):
    STOP = set("the a an and or but i to of in on for is are was were my it this that "
               "with you be have has had not do does did can will would should at as "
               "from by so no yes please thanks thank amazon".split())
    corpus_tf = Counter()
    doc_sets = []
    for t in texts:
        words = [w for w in re.findall(r"[a-z']+", t.lower()) if w not in STOP and len(w) > 2]
        doc_sets.append(words)
        corpus_tf.update(set(words))
    n_docs = len(texts)
    out = {}
    for cid in sorted(set(labels)):
        cl_tf = Counter()
        n_cl = 0
        for i, lab in enumerate(labels):
            if lab == cid:
                cl_tf.update(set(doc_sets[i]))
                n_cl += 1
        scores = {}
        for w, c in cl_tf.items():
            g = corpus_tf[w] / n_docs
            scores[w] = (c / max(n_cl, 1)) / (g + 1e-9) * np.log(1 + c)
        out[cid] = [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]
    return out


def summarize_clusters(texts, labels, vecs, keywords, examples_per=8):
    summary = {}
    for cid in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == cid]
        centroid = vecs[idx].mean(axis=0)
        idx_sorted = sorted(idx, key=lambda i: np.linalg.norm(vecs[i] - centroid))
        summary[cid] = {
            "size": len(idx),
            "pct": round(len(idx) / len(labels) * 100, 1),
            "keywords": keywords[cid],
            "representative_messages": [texts[i] for i in idx_sorted[:examples_per]],
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--sample-n", type=int, default=3000)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--out", default="results/draft_taxonomy.json")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--full-label", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    print(f"Loaded {len(pairs)} pairs from {args.pairs}", file=sys.stderr)

    texts = sample_messages(pairs, args.sample_n, seed=args.seed)
    print(f"Sampled {len(texts)} customer messages for clustering", file=sys.stderr)

    vecs = embed(texts)
    best_k, km = choose_k(vecs, args.k_min, args.k_max, seed=args.seed)
    labels = km.labels_.tolist()
    keywords = cluster_keywords(texts, labels)
    summary = summarize_clusters(texts, labels, vecs, keywords)

    draft = {
        "brand": "AmazonHelp",
        "method": "sentence-transformers MiniLM + KMeans, k chosen by silhouette",
        "n_sampled": len(texts),
        "k_chosen": best_k,
        "clusters": {
            str(cid): {
                "proposed_intent_name": None,
                "size": s["size"],
                "pct_of_sample": s["pct"],
                "top_keywords": s["keywords"],
                "representative_messages": s["representative_messages"],
            }
            for cid, s in summary.items()
        },
    }

    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            label_map = json.load(f)
        renamed = {}
        for cid, info in draft["clusters"].items():
            name = label_map.get(cid) or label_map.get(int(cid)) or f"unlabeled_cluster_{cid}"
            renamed[name] = info
            renamed[name]["proposed_intent_name"] = name
        draft["clusters"] = renamed
        draft["labels_locked"] = True

        if args.full_label:
            print("Embedding full corpus for intent assignment ...", file=sys.stderr)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            all_texts = [p["customer_text"] for p in pairs]
            cent = {}
            for cid in set(labels):
                idx = [i for i, l in enumerate(labels) if l == cid]
                name = label_map.get(str(cid)) or label_map.get(int(cid)) or f"cluster_{cid}"
                cent[name] = vecs[idx].mean(axis=0)
            names = list(cent.keys())
            mat = np.vstack([cent[n] for n in names])
            mat /= np.linalg.norm(mat, axis=1, keepdims=True)

            assigned = Counter()
            B = 1024
            for i in range(0, len(all_texts), B):
                v = model.encode(all_texts[i:i + B], batch_size=B)
                nn = np.argmax(v @ mat.T, axis=1)
                assigned.update(names[j] for j in nn)
            draft["full_corpus_intent_distribution"] = dict(assigned.most_common())
            draft["full_corpus_n"] = len(all_texts)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(draft, f, indent=2, ensure_ascii=False)
    print(f"Taxonomy draft written to {args.out}", file=sys.stderr)

    print("\n" + "=" * 90)
    for cid, s in summary.items():
        print(f"\nCluster {cid}  (n={s['size']}, {s['pct']}% of sample)")
        print(f"  Keywords: {', '.join(s['keywords'])}")
        print("  Examples:")
        for m in s["representative_messages"][:4]:
            print(f"    - {m[:130]}")
    print("=" * 90)


if __name__ == "__main__":
    main()
