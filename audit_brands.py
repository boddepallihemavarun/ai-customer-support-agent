"""
Data Audit Script — Customer Support on Twitter Dataset
=========================================================
Purpose: Before committing to a brand (FR-02) or assuming grounding is
feasible (Section 15), inspect real data quality for several candidate
brands. This answers:

  1. How much conversation VOLUME does each candidate brand have?
  2. How many threads can actually be RECONSTRUCTED
     (customer -> brand reply -> possibly further customer reply)?
  3. How much of the brand's replies are BOILERPLATE
     ("please DM us", "sorry to hear that") vs SUBSTANTIVE
     (contains concrete info: numbers, policies, timeframes, links)?
  4. What does the apparent intent/topic spread look like (rough proxy
     via keyword buckets), to sanity check "6-12 intents" is plausible.

Usage:
    python audit_brands.py --csv /path/to/twcs.csv --brands AmazonHelp AppleSupport Delta SpotifyCares

Output:
    Prints a comparison table and writes results/brand_audit.json
    with the numbers needed to justify Decision Log #1 (brand selection).
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Language filtering
# ---------------------------------------------------------------------------
# We try to use langdetect if it's installed (pip install langdetect), since
# it's far more reliable than a heuristic. If it's not available, we fall
# back to a simple ASCII-ratio heuristic: real English tweets are almost
# entirely ASCII (allowing for the occasional emoji/accented word), while
# Japanese/Korean/Arabic/etc. text will have a low ASCII ratio.
# This is a KNOWN LIMITATION worth noting in the decision log: heuristic
# language filtering is not perfect and may misclassify some short or
# emoji-heavy English tweets, or short non-English tweets that happen to be
# mostly ASCII (e.g. Vietnamese without diacritics, or Indonesian).
try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0  # deterministic results
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False


def is_english(text: str, min_ascii_ratio: float = 0.85, min_len: int = 6) -> bool:
    """Return True if text is judged to be English. See module docstring
    above for method and limitations."""
    if not isinstance(text, str) or len(text.strip()) < min_len:
        return False  # too short to judge reliably; treated as non-English/excluded

    if _HAS_LANGDETECT:
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    # Fallback heuristic
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / len(text)
    return ratio >= min_ascii_ratio

# ---------------------------------------------------------------------------
# Heuristics for "boilerplate" detection.
# These are intentionally simple and are meant only to give a DIRECTIONAL
# signal, not a ground truth label. Document this as a known limitation.
# ---------------------------------------------------------------------------
BOILERPLATE_PATTERNS = [
    r"\bdm us\b",
    r"\bdm me\b",
    r"\bsend us a dm\b",
    r"\bplease dm\b",
    r"\bsorry to hear\b",
    r"\bwe('|’)?re sorry\b",
    r"\bwe apologize\b",
    r"\bcan you (please )?dm\b",
    r"\bfollow (us|and dm)\b",
    r"\breach out to us\b",
    r"\bwe('|’)?d like to (help|look into)\b",
    r"\bplease send (us )?your\b",
]

SUBSTANTIVE_SIGNALS = [
    r"\d+\s*(business )?days?\b",
    r"\d+\s*hours?\b",
    r"\$\d+",
    r"\bhttps?://\S+\b",
    r"\border (number|id) is\b",
    r"\brefund(ed)?\b",
    r"\btracking\b",
    r"\bcase (number|id)\b",
]

BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)
SUBSTANTIVE_RE = re.compile("|".join(SUBSTANTIVE_SIGNALS), re.IGNORECASE)

# Rough intent-proxy keyword buckets — NOT the final taxonomy (Section 11
# requires taxonomy to be *derived* from data), just a sanity-check signal
# that recognizable, separable issue clusters exist at all.
INTENT_PROXY_KEYWORDS = {
    "delivery_delay": [r"\bdelay(ed)?\b", r"\bhasn'?t arrived\b", r"\bstill (haven'?t|not) (received|got)\b", r"\bwhere is my (order|package)\b"],
    "order_status": [r"\border status\b", r"\btrack(ing)?\b", r"\bwhere('|s| is)? my order\b"],
    "refund_request": [r"\brefund\b", r"\bmoney back\b", r"\breimburse\b"],
    "payment_problem": [r"\bcharged\b", r"\bpayment (failed|issue|declined)\b", r"\bbilling\b", r"\bdouble charge\b"],
    "cancellation": [r"\bcancel(led|lation)?\b"],
    "account_issue": [r"\bcan'?t log ?in\b", r"\baccount (locked|suspended|disabled)\b", r"\bpassword\b"],
    "technical_problem": [r"\bnot working\b", r"\bcrash(ed|ing)?\b", r"\bbug\b", r"\berror\b", r"\bbroken\b"],
    "security_fraud": [r"\bhacked\b", r"\bfraud(ulent)?\b", r"\bunauthorized\b", r"\bscam\b"],
}
INTENT_PROXY_RE = {k: re.compile("|".join(v), re.IGNORECASE) for k, v in INTENT_PROXY_KEYWORDS.items()}


def classify_boilerplate(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    has_boiler = bool(BOILERPLATE_RE.search(text))
    has_subst = bool(SUBSTANTIVE_RE.search(text))
    if has_subst and not has_boiler:
        return "substantive"
    if has_boiler and not has_subst:
        return "boilerplate"
    if has_boiler and has_subst:
        return "mixed"
    return "other"


def proxy_intents(text: str):
    if not isinstance(text, str):
        return []
    return [name for name, pat in INTENT_PROXY_RE.items() if pat.search(text)]


def audit_brand(df: pd.DataFrame, brand_handle: str, lang_filter: bool = False,
                 other_sample_n: int = 20, seed: int = 42,
                 lang_sample_n: int = 5000) -> dict:
    """
    df: full twcs dataframe with columns:
        tweet_id, author_id, inbound, created_at, text,
        response_tweet_id, in_response_to_tweet_id
    brand_handle: the author_id used by the brand's official account, e.g. 'AmazonHelp'
    lang_filter: if True, drop reconstructed pairs where either the customer
        or brand text is judged non-English (see is_english()).
    other_sample_n: how many "other" (unclassified-by-heuristic) brand
        replies to randomly sample for manual inspection.
    """
    brand_tweets = df[df["author_id"] == brand_handle]
    n_brand_tweets_raw = len(brand_tweets)

    if n_brand_tweets_raw == 0:
        return {"brand": brand_handle, "error": "No tweets found for this author_id"}

    # Reconstruct: for each brand reply, find the customer tweet it replied to.
    id_to_row = df.set_index("tweet_id")
    reconstructed_all = []
    for _, brow in brand_tweets.iterrows():
        parent_id = brow.get("in_response_to_tweet_id")
        if pd.isna(parent_id):
            continue
        try:
            parent_id = int(parent_id)
        except (ValueError, TypeError):
            continue
        if parent_id in id_to_row.index:
            prow = id_to_row.loc[parent_id]
            # prow could be a DataFrame if duplicate tweet_ids exist; guard it
            if isinstance(prow, pd.DataFrame):
                prow = prow.iloc[0]
            if prow.get("inbound") in (True, "True", "true"):
                reconstructed_all.append({
                    "customer_text": prow["text"],
                    "brand_text": brow["text"],
                })

    n_reconstructed_all = len(reconstructed_all)

    # Apply language filter if requested. For speed, language detection runs
    # on a random SAMPLE of pairs (default 5000) rather than the full set —
    # langdetect is slow (pure-Python, per-call model inference) and calling
    # it on every one of 100k+ pairs can take tens of minutes per brand. A
    # few thousand pairs gives a statistically solid estimate of the English
    # fraction and downstream boilerplate/intent percentages for an audit;
    # it is NOT meant to produce the final production-cleaned dataset (that
    # comes later, in the Step 2 pipeline, where we can parallelize or use a
    # faster detector).
    n_dropped_non_english = 0
    n_analyzed_for_lang = n_reconstructed_all
    if lang_filter:
        rng_lang = random.Random(seed)
        pool = reconstructed_all
        if len(pool) > lang_sample_n:
            pool = rng_lang.sample(reconstructed_all, lang_sample_n)
        n_analyzed_for_lang = len(pool)
        reconstructed = []
        for r in pool:
            if is_english(r["customer_text"]) and is_english(r["brand_text"]):
                reconstructed.append(r)
            else:
                n_dropped_non_english += 1
    else:
        reconstructed = reconstructed_all

    n_reconstructed = len(reconstructed)
    n_brand_tweets = n_brand_tweets_raw  # denominator for reconstruction rate stays on raw volume
    reconstruction_rate = n_reconstructed_all / n_brand_tweets if n_brand_tweets else 0

    # Boilerplate vs substantive on the brand side of reconstructed (post-filter) pairs
    boiler_labels = [classify_boilerplate(r["brand_text"]) for r in reconstructed]
    boiler_counts = Counter(boiler_labels)
    total_r = max(n_reconstructed, 1)
    boiler_pct = {k: round(v / total_r * 100, 1) for k, v in boiler_counts.items()}

    # Sample "other" bucket brand replies for manual eyeballing
    rng = random.Random(seed)
    other_pairs = [r for r, lbl in zip(reconstructed, boiler_labels) if lbl == "other"]
    other_sample = rng.sample(other_pairs, min(other_sample_n, len(other_pairs)))

    # Proxy intent coverage on the customer side
    intent_hits = Counter()
    multi_intent_count = 0
    no_intent_count = 0
    for r in reconstructed:
        hits = proxy_intents(r["customer_text"])
        if len(hits) > 1:
            multi_intent_count += 1
        if len(hits) == 0:
            no_intent_count += 1
        for h in hits:
            intent_hits[h] += 1

    return {
        "brand": brand_handle,
        "lang_filter_applied": lang_filter,
        "lang_detect_method": "langdetect" if _HAS_LANGDETECT else "ascii_heuristic",
        "lang_filter_sampled": lang_filter and n_analyzed_for_lang < n_reconstructed_all,
        "n_pairs_analyzed_for_lang_filter": n_analyzed_for_lang,
        "n_brand_tweets": n_brand_tweets,
        "n_reconstructed_pairs_before_lang_filter": n_reconstructed_all,
        "n_reconstructed_pairs_after_lang_filter": n_reconstructed,
        "n_dropped_non_english_pairs": n_dropped_non_english,
        "estimated_pct_english_pairs": round((n_analyzed_for_lang - n_dropped_non_english) / max(n_analyzed_for_lang, 1) * 100, 1) if lang_filter else None,
        "reconstruction_rate_pct": round(reconstruction_rate * 100, 1),
        "brand_reply_boilerplate_pct": boiler_pct,
        "other_bucket_pct_of_total": round(len(other_pairs) / total_r * 100, 1),
        "other_bucket_sample_for_manual_review": other_sample,
        "proxy_intent_distribution": dict(intent_hits.most_common()),
        "pct_customer_msgs_matching_no_known_proxy_intent": round(no_intent_count / total_r * 100, 1),
        "pct_customer_msgs_matching_multiple_proxy_intents": round(multi_intent_count / total_r * 100, 1),
        "sample_reconstructed_pairs": reconstructed[:3],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit candidate brands in the Twitter customer support dataset")
    parser.add_argument("--csv", required=True, help="Path to twcs.csv")
    parser.add_argument("--brands", nargs="+", required=True, help="List of author_id handles to audit, e.g. AmazonHelp AppleSupport")
    parser.add_argument("--out", default="results/brand_audit.json")
    parser.add_argument("--lang-filter", action="store_true",
                         help="Drop pairs where customer or brand text is judged non-English. "
                              "Uses langdetect if installed (pip install langdetect), else an "
                              "ASCII-ratio heuristic.")
    parser.add_argument("--other-sample-n", type=int, default=20,
                         help="How many 'other' (unclassified) brand replies to sample per brand "
                              "for manual inspection.")
    parser.add_argument("--lang-sample-n", type=int, default=5000,
                         help="Max number of reconstructed pairs to run language detection on, "
                              "per brand (random sample). Keeps --lang-filter fast on large brands. "
                              "Set higher for a more precise estimate at the cost of speed.")
    args = parser.parse_args()

    if not args.lang_filter:
        print("NOTE: running WITHOUT --lang-filter. Pass --lang-filter to filter to English-only "
              "pairs (recommended before finalizing brand choice).", file=sys.stderr)
    elif not _HAS_LANGDETECT:
        print("NOTE: 'langdetect' not installed, falling back to ASCII-ratio heuristic for language "
              "filtering. For more accurate results: pip install langdetect", file=sys.stderr)

    print(f"Loading {args.csv} ...", file=sys.stderr)
    df = pd.read_csv(args.csv)

    # Basic schema check
    expected_cols = {"tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"WARNING: expected columns missing: {missing}. Adjust column names in script.", file=sys.stderr)

    if args.lang_filter:
        print(f"Language filter ON: sampling up to {args.lang_sample_n} pairs per brand "
              f"for language detection (keeps this fast). Use --lang-sample-n to change.",
              file=sys.stderr)

    results = []
    for brand in args.brands:
        print(f"Auditing {brand} ... (this may take a moment if --lang-filter is on)", file=sys.stderr)
        results.append(audit_brand(df, brand, lang_filter=args.lang_filter,
                                    other_sample_n=args.other_sample_n,
                                    lang_sample_n=args.lang_sample_n))
        print(f"  done with {brand}.", file=sys.stderr)

    # Print comparison table
    print("\n" + "=" * 115)
    print(f"{'Brand':<16} {'#Tweets':>9} {'#Recon(raw)':>12} {'#Recon(en)':>11} {'Recon %':>9} "
          f"{'Subst %':>9} {'Boiler %':>9} {'Other %':>9}")
    print("=" * 115)
    for r in results:
        if "error" in r:
            print(f"{r['brand']:<16} ERROR: {r['error']}")
            continue
        subst = r["brand_reply_boilerplate_pct"].get("substantive", 0)
        boiler = r["brand_reply_boilerplate_pct"].get("boilerplate", 0)
        other = r["other_bucket_pct_of_total"]
        print(f"{r['brand']:<16} {r['n_brand_tweets']:>9} {r['n_reconstructed_pairs_before_lang_filter']:>12} "
              f"{r['n_reconstructed_pairs_after_lang_filter']:>11} {r['reconstruction_rate_pct']:>8}% "
              f"{subst:>8}% {boiler:>8}% {other:>8}%")
    print("=" * 115)
    print("\n'Other' bucket samples for manual review (per brand, in the JSON output under "
          "'other_bucket_sample_for_manual_review'): eyeball these to confirm they're genuinely "
          "informative/neutral text and not boilerplate my regex missed.\n")
    print("\nInterpretation guide:")
    print(" - Reconstruction rate: % of brand replies where the customer message they")
    print("   replied to could be found. Low rate = threading is broken / hard to use.")
    print(" - Substantive %: brand replies containing concrete info (numbers, refund")
    print("   mentions, tracking, links). Higher is better for grounding (Section 15).")
    print(" - Boilerplate %: generic 'please DM us' style replies. High boilerplate")
    print("   means retrieval will mostly retrieve near-duplicate uninformative text —")
    print("   a candidate 'misleading headline number' risk (Section 28).")
    print(" - Check 'proxy_intent_distribution' per brand in the JSON output to sanity")
    print("   check that 6-12 separable intents (Section 11) are plausible.\n")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
