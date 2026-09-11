import json

rows = [json.loads(l) for l in open("results/golden_set_labeled.jsonl", encoding="utf-8")]

INTENTS = ["delivery_issue", "damaged_item_or_product_issue",
           "prime_membership_issue", "support_process_complaint", "other"]

def prf(pred, true, cls):
    tp = sum(1 for p, t in zip(pred, true) if p == cls and t == cls)
    fp = sum(1 for p, t in zip(pred, true) if p == cls and t != cls)
    fn = sum(1 for p, t in zip(pred, true) if p != cls and t == cls)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

pred_i = [r["pred_intent"] for r in rows]
true_i = [r["true_intent"] for r in rows]

print(f"n = {len(rows)}")
print(f"\n{'intent':32s} {'prec':>6s} {'rec':>6s} {'f1':>6s}  support")
for c in INTENTS:
    p, r_, f1 = prf(pred_i, true_i, c)
    print(f"{c:32s} {p:6.2f} {r_:6.2f} {f1:6.2f}  {true_i.count(c)}")

acc = sum(p == t for p, t in zip(pred_i, true_i)) / len(rows)
print(f"\nIntent accuracy: {acc:.2f}")

mism = [(i, p, t) for i, (p, t) in enumerate(zip(pred_i, true_i)) if p != t]
print(f"\nMismatches: {len(mism)}")
for i, p, t in mism[:15]:
    print(f"  row {i}: pred={p}  true={t}")
