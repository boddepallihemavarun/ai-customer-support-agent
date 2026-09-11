"""Sample 200 messages stratified by predicted intent for hand-labeling."""
import json
import random
import sys
from collections import Counter

from agent_pipeline import build_centroids, classify
from sentence_transformers import SentenceTransformer

pairs = [json.loads(l) for l in open("data/amazon_pairs.jsonl", encoding="utf-8")]
random.Random(42).shuffle(pairs)

model = SentenceTransformer("all-MiniLM-L6-v2")
cent = build_centroids("results/draft_taxonomy.json", model)

per_class = 50
counts = Counter()
out = []
for p in pairs:
    if len(out) >= per_class * len(cent):
        break
    intent, sim, _ = classify(p["customer_text"], model, cent)
    if counts[intent] < per_class:
        counts[intent] += 1
        out.append({"customer_text": p["customer_text"],
                    "pred_intent": intent,
                    "true_intent": None,
                    "should_escalate": None})

with open("results/golden_set_unlabeled.jsonl", "w", encoding="utf-8") as f:
    for item in out:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Wrote {len(out)} items: {dict(counts)}", file=sys.stderr)
