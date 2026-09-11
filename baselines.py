"""Two baselines vs the agent, evaluated on the same golden set.
Usage: python baselines.py
"""
import json
import numpy as np
from collections import Counter

rows = [json.loads(l) for l in open("results/golden_set_labeled.jsonl", encoding="utf-8")]
texts = [r["customer_text"] for r in rows]
true_i = [r["true_intent"] for r in rows]

def accuracy(preds): return sum(p == t for p, t in zip(preds, true_i)) / len(rows)

def macro_f1(preds):
    labels = sorted(set(true_i))
    f1s = []
    for lab in labels:
        tp = sum(1 for p, t in zip(preds, true_i) if p == lab and t == lab)
        fp = sum(1 for p, t in zip(preds, true_i) if p == lab and t != lab)
        fn = sum(1 for p, t in zip(preds, true_i) if p != lab and t == lab)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    return float(np.mean(f1s))

# --- Baseline 1: majority class (predict the most frequent true intent) ---
majority = Counter(true_i).most_common(1)[0][0]
preds_majority = [majority] * len(rows)
print(f"Majority baseline (always '{majority}'): "
      f"acc={accuracy(preds_majority):.2f} macroF1={macro_f1(preds_majority):.2f}")

# --- Baseline 2: TF-IDF + Logistic Regression, 5-fold CV on the golden set ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = vec.fit_transform(texts)
clf = LogisticRegression(max_iter=1000, C=1.0)
preds_tfidf = cross_val_predict(clf, X, np.array(true_i), cv=5)
print(f"TF-IDF + LogReg (5-fold CV):      "
      f"acc={accuracy(preds_tfidf):.2f} macroF1={macro_f1(preds_tfidf):.2f}")

# Agent number for the comparison table (from evaluate_agent.py)
preds_agent = [r["pred_intent"] for r in rows]
print(f"Agent (centroid + other-fallback): "
      f"acc={accuracy(preds_agent):.2f} macroF1={macro_f1(preds_agent):.2f}")
