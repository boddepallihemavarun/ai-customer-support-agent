"""Rescore golden set: centroid classifier + 'other' confidence fallback."""
import json
import sys

sys.path.insert(0, ".")
from agent_pipeline import build_centroids, classify

OTHER_THRESH = 0.30

rows = [json.loads(l) for l in open("results/golden_set_labeled.jsonl", encoding="utf-8")]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
cent = build_centroids("results/draft_taxonomy.json", model)

n_other = 0
for r in rows:
    intent, sim, sims = classify(r["customer_text"], model, cent)
    if max(sims.values()) < OTHER_THRESH:
        r["pred_intent"] = "other"
        n_other += 1
    else:
        r["pred_intent"] = intent

with open("results/golden_set_labeled.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Rescored {len(rows)} rows; 'other' fallback fired {n_other} times")
