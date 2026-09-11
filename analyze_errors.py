import json
from collections import defaultdict

rows = [json.loads(l) for l in open("results/golden_set_labeled.jsonl", encoding="utf-8")]

groups = defaultdict(list)
for i, r in enumerate(rows):
    if r["pred_intent"] != r["true_intent"]:
        groups[(r["pred_intent"], r["true_intent"])].append((i, r["customer_text"]))

for (pred, true), items in sorted(groups.items(), key=lambda x: -len(x[1])):
    print(f"\n{'='*70}")
    print(f"PRED={pred}  ->  TRUE={true}   ({len(items)} cases)")
    print(f"{'='*70}")
    for i, txt in items:
        print(f"  [{i}] {txt[:100]}")
