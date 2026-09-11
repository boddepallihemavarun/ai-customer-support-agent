"""
Pipeline — Customer Support on Twitter (AmazonHelp)
Step 2: Production data cleaning & thread reconstruction.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

BRAND = "AmazonHelp"

PART_MARKER_RE = re.compile(r"\(?\s*(\d+)\s*/\s*(\d+)\s*\)?")
AGENT_SIG_RE = re.compile(r"\s*\^[A-Z]{1,3}\s*$")
SLASH_SIG_RE = re.compile(r"\s*/[A-Z]{1,2}\s*$")
STAR_SIG_RE = re.compile(r"\s*\*[A-Z]{2,3}\s*$")
MASKED_HANDLE_RE = re.compile(r"@\d{3,}")
WS_RE = re.compile(r"\s+")
PRIVACY_RE = re.compile(
    r"(don'?t (provide|share) your (order|account|payment)|"
    r"personal information|visible to (the )?public)",
    re.IGNORECASE,
)


def strip_signatures(text):
    if not isinstance(text, str):
        return ""
    t = text
    t = AGENT_SIG_RE.sub(" ", t)
    t = SLASH_SIG_RE.sub(" ", t)
    t = STAR_SIG_RE.sub(" ", t)
    t = re.sub(r"\(\s*\d+\s*/\s*\d+\s*\)", " ", t)
    t = re.sub(r"^\s*\d+\s*/\s*\d+\s+", " ", t)
    t = re.sub(r"\s+\d+\s*/\s*\d+\s*$", " ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def clean_text(text):
    if not isinstance(text, str):
        return ""
    t = MASKED_HANDLE_RE.sub("", text)
    t = WS_RE.sub(" ", t).strip()
    return t


def is_english(text, min_len=6):
    if not isinstance(text, str) or len(text.strip()) < min_len:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def parse_part_marker(text):
    paren = re.search(r"\(\s*(\d+)\s*/\s*(\d+)\s*\)", text)
    if paren:
        return int(paren.group(1)), int(paren.group(2))
    bare = PART_MARKER_RE.search(text)
    if bare:
        p, t = int(bare.group(1)), int(bare.group(2))
        if 1 <= p <= t <= 9:
            return p, t
    return None


def stitch_multipart(brand_rows_by_parent):
    stitched = {}
    stats = {"parent_with_multipart": 0, "parent_with_multiple_replies": 0}
    for parent_id, rows in brand_rows_by_parent.items():
        if len(rows) == 1:
            stitched[parent_id] = [(1, 1, {"text": rows[0]["text"],
                                           "tweet_ids": [rows[0]["tweet_id"]]})]
            continue

        parts = []
        singles = []
        for r in rows:
            marker = parse_part_marker(r["text"])
            if marker:
                parts.append((marker[0], marker[1], r))
            else:
                singles.append(r)

        entries = []
        if parts:
            by_total = defaultdict(dict)
            for p, t, r in parts:
                by_total[t][p] = r
            merged_groups = []
            for t, pmap in by_total.items():
                if set(pmap.keys()) == set(range(1, t + 1)):
                    merged_groups.append([pmap[i] for i in range(1, t + 1)])
                else:
                    for p, r in pmap.items():
                        singles.append(r)
            for group in merged_groups:
                stats["parent_with_multipart"] += 1
                entries.append((1, 1, {
                    "text": " ".join(r["text"] for r in group),
                    "tweet_ids": [r["tweet_id"] for r in group],
                }))

        if singles:
            stats["parent_with_multiple_replies"] += 1
            for r in singles:
                entries.append((1, 1, {"text": r["text"],
                                       "tweet_ids": [r["tweet_id"]]}))

        stitched[parent_id] = entries
    return stitched, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-pairs", default="data/amazon_pairs.jsonl")
    parser.add_argument("--out-threads", default="data/amazon_threads.jsonl")
    parser.add_argument("--out-provenance", default="results/amazon_pipeline_provenance.json")
    parser.add_argument("--flag-privacy", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...", file=sys.stderr)
    df = pd.read_csv(args.csv)

    prov = {"brand": BRAND, "steps": []}

    def log_step(name, **kw):
        prov["steps"].append({"step": name, **kw})
        print(f"  [{name}] " + " ".join(f"{k}={v}" for k, v in kw.items()), file=sys.stderr)

    log_step("load", total_tweets=len(df))

    brand_tweets = df[df["author_id"] == BRAND]
    log_step("brand_filter", brand_tweets=len(brand_tweets))

    id_to_row = df.set_index("tweet_id")

    brand_by_parent = defaultdict(list)
    for _, brow in brand_tweets.iterrows():
        pid = brow.get("in_response_to_tweet_id")
        if pd.isna(pid):
            continue
        try:
            pid = int(pid)
        except (ValueError, TypeError):
            continue
        brand_by_parent[pid].append({"tweet_id": int(brow["tweet_id"]),
                                     "text": brow["text"]})

    stitched, stitch_stats = stitch_multipart(brand_by_parent)
    log_step("stitch_multipart", **stitch_stats,
             n_parents_with_brand_reply=len(stitched))

    raw_pairs = []
    for pid, entries in stitched.items():
        if pid not in id_to_row.index:
            continue
        prow = id_to_row.loc[pid]
        if isinstance(prow, pd.DataFrame):
            prow = prow.iloc[0]
        if prow.get("inbound") not in (True, "True"):
            continue
        for _, _, entry in entries:
            raw_pairs.append({
                "customer_tweet_id": int(pid),
                "brand_tweet_ids": entry["tweet_ids"],
                "customer_text_raw": prow["text"],
                "brand_text_raw": entry["text"],
            })
    log_step("pairs_raw", n_pairs=len(raw_pairs))

    for p in raw_pairs:
        p["customer_text"] = clean_text(p["customer_text_raw"])
        p["brand_text"] = strip_signatures(p["brand_text_raw"])

    kept = []
    n_dropped = 0
    for p in raw_pairs:
        if is_english(p["customer_text"]) and is_english(p["brand_text"]):
            kept.append(p)
        else:
            n_dropped += 1
    log_step("language_filter", kept=len(kept), dropped_non_english=n_dropped,
             pct_english=round(len(kept) / max(len(raw_pairs), 1) * 100, 1))

    if args.flag_privacy:
        PRIVACY_RE_ = re.compile(
            r"(don'?t (provide|share) your (order|account|payment)|"
            r"personal information|visible to (the )?public)", re.IGNORECASE)
        for p in kept:
            p["privacy_boilerplate"] = bool(PRIVACY_RE_.search(p["brand_text"]))
        log_step("privacy_flag", flagged=sum(p["privacy_boilerplate"] for p in kept))

    final_pairs = [p for p in kept if p["customer_text"] and p["brand_text"]]
    log_step("final_pairs", n=len(final_pairs),
             dropped_empty=len(kept) - len(final_pairs))

    os.makedirs(os.path.dirname(args.out_pairs) or ".", exist_ok=True)
    with open(args.out_pairs, "w") as f:
        for p in final_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    text_by_id = df.set_index("tweet_id")["text"].to_dict()
    inbound_by_id = df.set_index("tweet_id")["inbound"].to_dict()
    resp_by_id = df.set_index("tweet_id")["response_tweet_id"].to_dict()

    def to_int_list(x):
        if pd.isna(x):
            return []
        return [int(v) for v in str(x).replace(",", " ").split()]

    threads = []
    for pid in stitched.keys():
        if pid not in text_by_id:
            continue
        if str(inbound_by_id.get(pid, "")).lower() != "true":
            continue
        chain = [{"author": "customer", "tweet_id": pid, "text": text_by_id[pid]}]
        cur = pid
        seen = {pid}
        while len(chain) < 10:
            nxt_ids = [i for i in to_int_list(resp_by_id.get(cur)) if i not in seen]
            if not nxt_ids:
                break
            cur = nxt_ids[0]
            seen.add(cur)
            author = "customer" if str(inbound_by_id.get(cur, "")).lower() == "true" else "brand"
            chain.append({"author": author, "tweet_id": cur,
                          "text": text_by_id.get(cur, "")})
        if len(chain) >= 2:
            threads.append(chain)

    for th in threads:
        for t in th:
            t["text_clean"] = (clean_text(t["text"])
                               if t["author"] == "customer"
                               else strip_signatures(t["text"]))
    with open(args.out_threads, "w") as f:
        for th in threads:
            f.write(json.dumps(th, ensure_ascii=False) + "\n")
    log_step("threads", n_threads=len(threads),
             avg_turns=round(sum(len(t) for t in threads) / max(len(threads), 1), 2))

    prov["outputs"] = {"pairs": args.out_pairs, "threads": args.out_threads}
    os.makedirs(os.path.dirname(args.out_provenance) or ".", exist_ok=True)
    with open(args.out_provenance, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Provenance written to {args.out_provenance}", file=sys.stderr)


if __name__ == "__main__":
    main()
