import json, re

IN = "results/golden_set_unlabeled.jsonl"
OUT = "results/golden_set_prelabel.jsonl"

ESC_PATTERNS = [
    (r"\b(lawsuit|legal action|lawyer|attorney|court)\b", "legal threat"),
    (r"\b(hacked|fraud|fraudsters?|scam|scammed|cheat(ed)?|stole|stolen|unauthorized|unauthorised)\b", "fraud allegation"),
    (r"\b(charge[sd]?|refund(ed)?|my money|money back|dispute)\b", "money at stake"),
    (r"\b(data protection|privacy|breach(ed)?)\b", "privacy claim"),
    (r"\b(wtf|fuck|shit|damn|pathetic|disgusting|atrocious|scum)\b", "hostile language"),
    (r"(3 days in a row|4 days|5 weeks|22 days|45 days|2 months|10 ?days|every single day|third time|3rd time)", "repeated failure"),
    (r"\b(lie|lying|lied)\b", "dishonesty allegation"),
]

def guess_intent(t):
    tl = t.lower()
    if re.search(r"\b(prime (membership|video|photos|now|subscription)|membership|subscription|free trial)\b", tl) \
       and not re.search(r"\b(package|parcel|order|deliver\w*)\b", tl):
        return "prime_membership_issue"
    if re.search(r"\b(deliver\w*|package|parcel|shipment|shipped|tracking|carrier|driver|courier|wrong address|late)\b", tl):
        return "delivery_issue"
    if re.search(r"\b(damaged|broken|defective|fake|wrong item|return|refund|replace\w*|not working|quality)\b", tl):
        return "damaged_item_or_product_issue"
    if re.search(r"\b(customer (service|care|support)|executive|no (reply|response|help)|chat|called|thanks|thank you|resolved|form)\b", tl):
        return "support_process_complaint"
    return "other"

rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
out = open(OUT, "w", encoding="utf-8")
for r in rows:
    t = r["customer_text"]
    hits = [name for pat, name in ESC_PATTERNS if re.search(pat, t.lower())]
    r["true_intent"] = guess_intent(t)
    r["should_escalate"] = len(hits) > 0
    r["prelabel_reason"] = ", ".join(hits) if hits else "none"
    out.write(json.dumps(r, ensure_ascii=False) + "\n")
out.close()
print("Wrote", OUT, "-", len(rows), "items")
