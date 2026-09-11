"""
Compare human overall scores with LLM-judge helpfulness scores.

Workflow:
  1. python human_agreement.py --template
  2. Open results/human_agreement_scores.json
  3. Fill "overall" (1-5) for 30-50 examples after human review.
  4. python human_agreement.py

The human scores must be genuine human ratings; do not fabricate them.
"""

import argparse
import json
from pathlib import Path

from scipy.stats import spearmanr

JUDGE_PATH = Path("results/judge_results.json")
HUMAN_PATH = Path("results/human_agreement_scores.json")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_template():
    if not JUDGE_PATH.exists():
        raise SystemExit(
            f"Missing {JUDGE_PATH}. Run `python llm_judge.py` first."
        )

    judge = load_json(JUDGE_PATH)

    # Use a deterministic 50-example audit set (or all if fewer than 50).
    selected = judge[:50]

    template = [
        {
            "customer_text": row["customer_text"],
            "overall": None,
        }
        for row in selected
    ]

    HUMAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HUMAN_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"Created {HUMAN_PATH} with {len(template)} examples.")
    print("Fill each `overall` field with a genuine human 1-5 rating, then rerun:")
    print("  python human_agreement.py")


def evaluate():
    if not JUDGE_PATH.exists():
        raise SystemExit("Missing results/judge_results.json. Run llm_judge.py first.")

    if not HUMAN_PATH.exists():
        print("Human score file does not exist.")
        print("Run: python human_agreement.py --template")
        raise SystemExit(1)

    judge = load_json(JUDGE_PATH)
    human = load_json(HUMAN_PATH)

    hmap = {}
    for row in human:
        text = row.get("customer_text")
        score = row.get("overall")
        if not text:
            continue
        if score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if not 1 <= score <= 5:
            raise SystemExit(
                f"Invalid human score {score!r} for: {text[:80]!r}. "
                "Use an integer from 1 to 5."
            )
        hmap[text] = score

    pairs = []
    for row in judge:
        text = row.get("customer_text")
        if text in hmap and row.get("helpfulness") is not None:
            pairs.append((hmap[text], float(row["helpfulness"])))

    if len(pairs) < 10:
        raise SystemExit(
            f"Only {len(pairs)} rated examples matched the judge results. "
            "Review at least 30 examples before reporting agreement."
        )

    h, j = zip(*pairs)
    rho, p = spearmanr(h, j)
    within_one = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)

    print(f"n={len(pairs)}")
    print(f"Spearman rho={rho:.2f} (p={p:.3f})")
    print(f"Within-1-point agreement={within_one:.0%}")

    if len(pairs) < 30:
        print(
            "WARNING: fewer than 30 human-rated examples. "
            "Use 30-50 for the assignment's human-vs-judge validation."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        action="store_true",
        help="Create a human-rating template from judge_results.json",
    )
    args = parser.parse_args()

    if args.template:
        make_template()
    else:
        evaluate()
