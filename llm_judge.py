"""LLM-as-judge for drafted customer-support replies.

Usage:
  set OPENROUTER_API_KEY=your_key
  python llm_judge.py

Optional:
  set OPENROUTER_MODEL=openai/gpt-chat-latest
  python llm_judge.py

The judge scores five dimensions from 1-5 and writes:
  results/judge_results.json

This is an independent evaluator: it sees the customer message, the drafted
reply, the predicted intent/route, and the retrieval-grounding metadata.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

INPUT = "results/agent_outputs.jsonl"
OUTPUT = "results/judge_results.json"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")

RUBRIC = {
    "relevance": "Does the reply directly address the customer's actual request/problem?",
    "helpfulness": "Does it provide a useful next step or resolution rather than a generic/unrelated statement?",
    "groundedness": "Is the reply consistent with the retrieved historical-support grounding and does it avoid unsupported specific claims?",
    "brand_consistency": "Does it sound appropriate for a professional Amazon customer-support reply: concise, empathetic, clear, and action-oriented?",
    "hallucination": "Does the reply avoid inventing policies, facts, guarantees, order details, or actions not supported by the customer message/grounding? 5 means no apparent unsupported claims.",
}

SYSTEM_PROMPT = """You are an independent evaluator for an Amazon customer-support AI assignment.

Evaluate ONLY the drafted reply. Do not rewrite it and do not reward a reply merely
because it is polite.

Score every dimension from 1 to 5:
1 = very poor, 2 = poor, 3 = mixed/acceptable, 4 = good, 5 = excellent.

Important:
- A generic reply that sounds professional but does not address the customer's issue
  should score low on relevance/helpfulness.
- If the reply is unrelated to the customer's message, relevance should normally be 1.
- Groundedness concerns consistency with the provided retrieval metadata. Do not assume
  that "grounded in past replies" proves the reply is appropriate to this customer.
- Hallucination means unsupported claims, promises, invented facts, or made-up actions.
- Be conservative and consistent.
Return ONLY valid JSON with this exact shape:
{
  "relevance": 1,
  "helpfulness": 1,
  "groundedness": 1,
  "brand_consistency": 1,
  "hallucination": 1,
  "overall": 1,
  "rationale": "brief explanation"
}
"""


def call_llm(customer_text, reply, intent, route, route_reason, grounding):
    user_prompt = f"""Customer message:
{customer_text}

Predicted intent:
{intent}

Agent route:
{route}
Route reason:
{route_reason}

Drafted reply:
{reply}

Retrieval grounding metadata:
{grounding}

Rubric:
{json.dumps(RUBRIC, indent=2)}

Score the reply now."""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
            "Content-Type": "application/json",
            "X-Title": "AI Customer Support Agent - Reply Judge",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))

    content = data["choices"][0]["message"]["content"]
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    scores = json.loads(content)

    required = [
        "relevance",
        "helpfulness",
        "groundedness",
        "brand_consistency",
        "hallucination",
        "overall",
    ]
    for key in required:
        value = scores.get(key)
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"Invalid {key}: {value}")

    return scores


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Set it first, then run: python llm_judge.py"
        )

    if not os.path.exists(INPUT):
        raise SystemExit(f"Missing {INPUT}. Run the agent pipeline first.")

    rows = [
        json.loads(line)
        for line in open(INPUT, encoding="utf-8")
        if line.strip()
    ]
    drafted = [r for r in rows if r.get("drafted_reply")]
    print(f"Judging {len(drafted)} drafted replies with {MODEL}")

    results = []
    failures = 0

    for i, r in enumerate(drafted, 1):
        try:
            scores = call_llm(
                r["customer_text"],
                r["drafted_reply"],
                r.get("predicted_intent", ""),
                r.get("route", ""),
                r.get("route_reason", ""),
                r.get("grounding", ""),
            )
            scores.update({
                "customer_text": r["customer_text"],
                "predicted_intent": r.get("predicted_intent"),
                "route": r.get("route"),
                "drafted_reply": r.get("drafted_reply"),
                "grounding": r.get("grounding"),
            })
            results.append(scores)
            print(f"{i}/{len(drafted)} done")
        except Exception as exc:
            failures += 1
            print(f"{i}/{len(drafted)} FAILED: {exc}")

        # Small pause to reduce burst/rate-limit problems.
        time.sleep(0.2)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {OUTPUT}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {failures}")

    if results:
        for key in [
            "relevance",
            "helpfulness",
            "groundedness",
            "brand_consistency",
            "hallucination",
            "overall",
        ]:
            vals = [x[key] for x in results]
            print(f"{key:>18s}: mean {sum(vals)/len(vals):.2f} / 5")


if __name__ == "__main__":
    main()
