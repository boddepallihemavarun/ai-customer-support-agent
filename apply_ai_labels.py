import json

L = [
("other",0),("delivery_issue",0),("prime_membership_issue",1),("support_process_complaint",0),
("delivery_issue",1),("other",0),("delivery_issue",1),("support_process_complaint",1),
("other",0),("delivery_issue",0),("delivery_issue",0),("support_process_complaint",1),
("damaged_item_or_product_issue",0),("support_process_complaint",0),("delivery_issue",0),
("delivery_issue",0),("damaged_item_or_product_issue",0),("delivery_issue",0),
("delivery_issue",0),("delivery_issue",0),("other",0),("support_process_complaint",1),
("delivery_issue",0),("delivery_issue",0),("damaged_item_or_product_issue",1),
("support_process_complaint",1),("support_process_complaint",1),("support_process_complaint",0),
("delivery_issue",0),("damaged_item_or_product_issue",0),("support_process_complaint",0),
("delivery_issue",0),("delivery_issue",0),("delivery_issue",0),("other",0),
("support_process_complaint",1),("support_process_complaint",0),("prime_membership_issue",0),
("support_process_complaint",1),("delivery_issue",0),("delivery_issue",0),
("support_process_complaint",0),("other",0),("other",0),("support_process_complaint",0),
("delivery_issue",0),("damaged_item_or_product_issue",0),("delivery_issue",0),
("support_process_complaint",0),("other",0),("delivery_issue",0),("other",1),
("delivery_issue",1),("prime_membership_issue",0),("damaged_item_or_product_issue",0),
("prime_membership_issue",1),("delivery_issue",0),("damaged_item_or_product_issue",1),
("delivery_issue",0),("delivery_issue",0),("other",0),("other",0),
("support_process_complaint",0),("support_process_complaint",0),("prime_membership_issue",0),
("prime_membership_issue",0),("delivery_issue",0),("damaged_item_or_product_issue",0),
("support_process_complaint",0),("delivery_issue",1),("delivery_issue",0),
("support_process_complaint",1),("delivery_issue",0),("support_process_complaint",0),
("delivery_issue",0),("support_process_complaint",0),("delivery_issue",0),
("delivery_issue",0),("delivery_issue",0),("delivery_issue",0),("delivery_issue",0),
("support_process_complaint",0),("delivery_issue",0),("other",0),("delivery_issue",0),
("support_process_complaint",0),("prime_membership_issue",1),("prime_membership_issue",0),
("other",0),("delivery_issue",0),("support_process_complaint",1),("support_process_complaint",0),
("other",0),("other",0),("delivery_issue",0),("support_process_complaint",0),
("other",0),("support_process_complaint",0),("damaged_item_or_product_issue",1),
("prime_membership_issue",0),("other",0),("delivery_issue",0),("other",0),
("support_process_complaint",1),("support_process_complaint",0),("other",0),
("damaged_item_or_product_issue",0),("support_process_complaint",0),("support_process_complaint",0),
("other",0),("delivery_issue",0),("other",0),("delivery_issue",1),
("delivery_issue",0),("delivery_issue",0),("support_process_complaint",1),("other",0),
("delivery_issue",0),("delivery_issue",0),("support_process_complaint",0),
("prime_membership_issue",0),("delivery_issue",0),("support_process_complaint",0),
("damaged_item_or_product_issue",0),("delivery_issue",0),("damaged_item_or_product_issue",0),
("delivery_issue",1),("other",0),("damaged_item_or_product_issue",1),
("delivery_issue",0),("delivery_issue",1),("other",0),("support_process_complaint",1),
("damaged_item_or_product_issue",1),("other",0),("other",0),("delivery_issue",0),
("other",0),("other",0),("delivery_issue",0),("damaged_item_or_product_issue",0),
("delivery_issue",0),("support_process_complaint",0),("support_process_complaint",0),
("support_process_complaint",0),("delivery_issue",0),("delivery_issue",0),
("support_process_complaint",0),("prime_membership_issue",0),("damaged_item_or_product_issue",1),
("damaged_item_or_product_issue",1),("other",0),("damaged_item_or_product_issue",1),
("damaged_item_or_product_issue",0),("prime_membership_issue",1),("prime_membership_issue",0),
("delivery_issue",0),("prime_membership_issue",0),("prime_membership_issue",0),
("prime_membership_issue",0),("other",0),("prime_membership_issue",0),
("prime_membership_issue",0),("other",0),("prime_membership_issue",0),
("support_process_complaint",0),("prime_membership_issue",0),("prime_membership_issue",0),
("prime_membership_issue",0),("prime_membership_issue",0),("prime_membership_issue",0),
("support_process_complaint",0),("prime_membership_issue",0),("prime_membership_issue",0),
("prime_membership_issue",0),("prime_membership_issue",0),("prime_membership_issue",0),
("prime_membership_issue",0),("damaged_item_or_product_issue",0),("prime_membership_issue",0),
("prime_membership_issue",0),("prime_membership_issue",0),("other",0),("other",0),
("prime_membership_issue",0),("prime_membership_issue",0),("prime_membership_issue",0),
("prime_membership_issue",0),("prime_membership_issue",1),("prime_membership_issue",1),
("prime_membership_issue",0),
]

rows = [json.loads(l) for l in open("results/golden_set_unlabeled.jsonl", encoding="utf-8")]
if len(L) < len(rows):
    print(f"WARNING: {len(rows)-len(L)} labels missing — padded; AUDIT rows {len(L)}..{len(rows)-1}")
    L += [("prime_membership_issue", 0)] * (len(rows) - len(L))

with open("results/golden_set_ai_labels.jsonl", "w", encoding="utf-8") as f:
    for r, (intent, esc) in zip(rows, L):
        f.write(json.dumps({"customer_text": r["customer_text"],
                            "ai_intent": intent, "ai_escalate": bool(esc),
                            "true_intent": None, "should_escalate": None},
                           ensure_ascii=False) + "\n")
print("Wrote results/golden_set_ai_labels.jsonl -", len(rows), "items")
print("Escalation rate:", sum(e for _, e in L) / len(L))
