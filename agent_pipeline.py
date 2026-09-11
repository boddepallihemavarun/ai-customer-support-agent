"""
Agent pipeline: classify -> draft reply -> auto-handle vs escalate.
Usage:
    python agent_pipeline.py --limit 500
"""

import argparse
import json
import random
import sys

import numpy as np

ESCALATION_TRIGGERS = [
    ("legal_threat",       ["lawsuit", "legal action", "sue ", "attorney", "lawyer"]),
    ("refund_high_stakes", ["refund", "money back", "chargeback", "dispute"]),
    ("account_security", ["hacked", "stolen", "fraud", "unauthorized", "password"]),
    ("urgent_timebound",   ["urgent", "emergency", "hospital", "deadline", "wedding", "birthday"]),
]

REPLY_MIN_SIM = 0.55
ESCALATE_CONF_THRESHOLD = 0.45

DELIVERY_HINTS = ["delivery", "delivered", "package", "parcel", "shipment",
                  "shipping", "order not received", "hasn't arrived",
                  "hasn't shipped", "not delivered", "lost", "late",
                  "out for delivery", "dispatch", "hasn't even shipped",
                  "hasn't got", "haven't received", "didn't show", "not show",
                  "pushed back", "swiped", "missing from"]
DEFECT_HINTS = ["damaged", "broken", "defective", "cracked", "dented",
                "leaked", "wrong item", "fake product", "not working", "torn",
                "counterfeit"]
PRODUCT_HINTS = ["item", "product", "order", "device", "unit", "box",
                 "phone", "tv", "book", "cable", "monitor"]
SUPPORT_HINTS = ["customer service", "customer support", "support team",
                 "no response", "no reply", "hang up", "hung up", "calling",
                 "executive", "chat", "contact", "customer care"]
REFUND_HINTS = ["refund", "cashback", "cash back", "money back", "cancel"]
PRAISE_HINTS = ["thank", "thanks", "resolved my issue", "appreciate",
                "great service", "love ", "shout out", "kudos", "fixed"]
QUESTION_HINTS = ["how can i", "how do i", "what is", "where can i",
                  "can you tell me", "is there a place", "can you block",
                  "how come", "what do i do", "may i ask why"]
COMPLAINT_HINTS = ["issue", "problem", "complaint", "worst", "terrible",
                   "cancel", "refund", "late", "not working", "sucks",
                   "bad", "disappointed", "upset", "angry", "frustrat",
                   "cheated", "fool", "ridiculous", "pathetic", "worse"]


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build_centroids(taxonomy_path, model):
    d = json.load(open(taxonomy_path, encoding="utf-8"))
    cent = {}
    for name, info in d["clusters"].items():
        msgs = info["representative_messages"]
        v = model.encode(msgs, normalize_embeddings=True)
        c = v.mean(axis=0)
        cent[name] = (c / np.linalg.norm(c)).astype(np.float32)
    return cent


def classify(text, model, cent):
    v = model.encode([text], normalize_embeddings=True)[0]
    names = list(cent.keys())
    sims = {n: float(v @ cent[n]) for n in names}
    best = max(sims, key=sims.get)
    return best, sims[best], sims


def has_any(low, kws):
    return any(k in low for k in kws)


def classify_with_rules(text, model, cent):
    intent, sim, all_sims = classify(text, model, cent)
    low = " " + text.lower() + " "

    has_delivery = has_any(low, DELIVERY_HINTS)
    has_defect = has_any(low, DEFECT_HINTS)
    has_product = has_any(low, PRODUCT_HINTS)
    has_support = has_any(low, SUPPORT_HINTS)
    has_refund = has_any(low, REFUND_HINTS)
    has_praise = has_any(low, PRAISE_HINTS)
    has_question = has_any(low, QUESTION_HINTS)
    has_complaint = has_any(low, COMPLAINT_HINTS)

    # Rule 1: praise/question with no complaint, no defect, no delivery -> other
    if (has_praise or has_question) and not has_complaint and not has_defect \
       and not has_delivery and not has_refund:
        return "other", sim, all_sims, "rule1: praise/question, no complaint"

    # Rule 2: delivery problem beats Prime mention
    if intent == "prime_membership_issue" and has_delivery and \
       has_any(low, ["lost", "late", "not show", "didn't show",
                     "haven't received", "not delivered", "hasn't arrived",
                     "stated as handed", "pushed back", "swiped",
                     "missing from", "out for delivery", "2 day", "two days"]):
        return "delivery_issue", sim, all_sims, "rule2: delivery beats prime"

    # Rule 3: damaged_item requires defect words; else -> support process
    if intent == "damaged_item_or_product_issue" and not has_defect:
        return "support_process_complaint", sim, all_sims, "rule3: no defect evidence"

    # Rule 4: underlying delivery problem beats support channel
    if intent == "support_process_complaint" and has_delivery and not has_support:
        return "delivery_issue", sim, all_sims, "rule4: delivery problem via channel"

    return intent, sim, all_sims, "centroid"


def decide_route(text, intent, sim):
    low = " " + text.lower() + " "
    hits = [name for name, kws in ESCALATION_TRIGGERS if any(k in low for k in kws)]
    if hits:
        return "escalate", f"escalation trigger(s): {', '.join(hits)}"
    if sim < ESCALATE_CONF_THRESHOLD:
        return "escalate", f"low intent confidence ({sim:.2f} < {ESCALATE_CONF_THRESHOLD})"
    return "auto_handle", f"intent={intent} (conf={sim:.2f}), no escalation triggers"


def draft_reply(text, vecs, meta, model, top=3):
    v = model.encode([text], normalize_embeddings=True)[0]
    sims = vecs @ v
    idx = np.argsort(-sims)[:top]
    grounded = [(meta[i]["brand_text"], float(sims[i])) for i in idx if sims[i] > REPLY_MIN_SIM]
    if not grounded:
        return None, f"no similar past case above {REPLY_MIN_SIM} (max {float(sims[idx[0]]):.2f})"
    best_reply, best_sim = grounded[0]
    return best_reply, f"grounded in {len(grounded)} past replies (best sim {best_sim:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/amazon_pairs.jsonl")
    ap.add_argument("--index", default="data/retrieval_index.npz")
    ap.add_argument("--meta", default="data/retrieval_meta.jsonl")
    ap.add_argument("--taxonomy", default="results/draft_taxonomy.json")
    ap.add_argument("--out", default="results/agent_outputs.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = load_jsonl(args.pairs)
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[:args.limit]
    print(f"Processing {len(pairs)} pairs", file=sys.stderr)

    vecs = np.load(args.index)["vecs"]
    meta = load_jsonl(args.meta)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cent = build_centroids(args.taxonomy, model)
    print(f"Centroids ready: {list(cent.keys())}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as out:
        for n, p in enumerate(pairs):
            text = p["customer_text"]
            intent, sim, all_sims, rule_reason = classify_with_rules(text, model, cent)
            route, reason = decide_route(text, intent, sim)
            reason = f"{reason} | {rule_reason}"
            reply, grounding = (None, "not drafted (escalated)")
            if route == "auto_handle":
                reply, grounding = draft_reply(text, vecs, meta, model)
            out.write(json.dumps({
                "customer_text": text,
                "predicted_intent": intent,
                "intent_confidence": round(sim, 3),
                "all_similarities": {k: round(v2, 3) for k, v2 in all_sims.items()},
                "route": route,
                "route_reason": reason,
                "drafted_reply": reply,
                "grounding": grounding,
            }, ensure_ascii=False) + "\n")
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{len(pairs)}", file=sys.stderr)
    print(f"Done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
