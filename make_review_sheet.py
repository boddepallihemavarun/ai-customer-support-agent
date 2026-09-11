import csv, json

rows = [json.loads(l) for l in open("results/golden_set_ai_labels.jsonl", encoding="utf-8")]
with open("results/review_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["row", "customer_text", "ai_intent", "ai_escalate", "true_intent", "should_escalate"])
    for i, r in enumerate(rows):
        w.writerow([i, r["customer_text"], r["ai_intent"], r["ai_escalate"],
                    r["ai_intent"], r["ai_escalate"]])
print("Wrote results/review_sheet.csv -", len(rows), "rows (E/F prefilled with AI labels)")
