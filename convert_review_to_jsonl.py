import csv, json

VALID_INTENTS = {"delivery_issue", "damaged_item_or_product_issue",
                 "prime_membership_issue", "support_process_complaint", "other"}
VALID_ESC = {"true", "false"}

orig = [json.loads(l) for l in open("results/golden_set_unlabeled.jsonl", encoding="utf-8")]
preds = {o["customer_text"]: o["pred_intent"] for o in orig}

rows = list(csv.DictReader(open("results/review_sheet.csv", encoding="utf-8-sig")))
out, errors = [], []
for r in rows:
    ti = (r["true_intent"] or "").strip().lower()
    se = (r["should_escalate"] or "").strip().lower()
    if ti not in VALID_INTENTS:
        errors.append(f'row {r["row"]}: bad true_intent "{ti}"')
        continue
    if se not in VALID_ESC:
        errors.append(f'row {r["row"]}: bad should_escalate "{se}"')
        continue
    i = int(r["row"])
    out.append({"customer_text": r["customer_text"],
                "pred_intent": preds.get(r["customer_text"], "UNKNOWN"),
                "true_intent": ti,
                "should_escalate": se == "true"})

if errors:
    print("FIX THESE BEFORE EXPORT:")
    for e in errors[:20]:
        print(" ", e)
    print(f"{len(errors)} bad rows - file NOT written")
else:
    with open("results/golden_set_labeled.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Wrote results/golden_set_labeled.jsonl - {len(out)} items, all labels valid")
