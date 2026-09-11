\# Customer Support Agent — Intent Classification Evaluation



\## Overview

Semantic intent classifier (sentence-transformer centroids) for a customer-support

triage agent, evaluated on a 200-item stratified golden set (seed 42).



\## Method

\- Golden set: 200 tweets, stratified by prelabel, seed 42

\- Label pipeline: rule-based prelabel → AI-assist → human audit

&#x20; (N labels overridden during audit)

\- Metrics: per-class precision/recall/F1, intent accuracy



\## Baseline results (pure centroid classifier)

| intent | prec | rec | f1 | support |

|---|---|---|---|---|

| delivery\_issue | 0.58 | 0.51 | 0.54 | 57 |

| damaged\_item\_or\_product\_issue | 0.22 | 0.55 | 0.31 | 20 |

| prime\_membership\_issue | 0.70 | 0.71 | 0.71 | 49 |

| support\_process\_complaint | 0.52 | 0.63 | 0.57 | 41 |

| other | 0.00 | 0.00 | 0.00 | 33 |

\*\*Accuracy: 0.51\*\* | `other` recall = 0.00 — the taxonomy had no "none of the above" route.



\## Error analysis

99/200 mismatches categorized:

\- \~33 true `other` forced into the 4 classes (no fallback route existed)

\- damaged\_item acting as catch-all for anger/refund tweets (precision 0.22)

\- Prime keyword hijacking delivery complaints (rows 28, 52, 139)

\- \~10 mismatches judged label ambiguity, not model error (audited; labels corrected)



\## Improvement experiments

1\. Keyword rule layer (delivery-beats-Prime, defect gates, praise→other): \*\*accuracy 0.46\*\* —

&#x20;  negative result. Hand-written keyword gates conflict with the semantic classifier;

&#x20;  defect keywords rarely matched real damaged-item phrasing (rule fired 45×, mostly wrongly).

2\. Confidence-threshold sweep (max centroid similarity < t → `other`): swept 12 configs,

&#x20;  measured against golden labels. Best: t=0.30 → `other` F1 0.00 → 0.20,

&#x20;  `other` precision 0.33. Damaged reroute rejected by the sweep (hurt accuracy at every setting).



\## Final configuration

Centroid classifier + low-confidence fallback to `other` (t = 0.30).

| intent | prec | rec | f1 |

|---|---|---|---|

| delivery\_issue | 0.60 | 0.50 | 0.55 |

| damaged\_item\_or\_product\_issue | 0.17 | 0.32 | 0.22 |

| prime\_membership\_issue | 0.70 | 0.74 | 0.72 |

| support\_process\_complaint | 0.53 | 0.58 | 0.56 |

| other | 0.33 | 0.14 | 0.20 |

\*\*Accuracy: 0.51\*\* — accuracy unchanged, but the agent can now route off-topic traffic

(previously 0% recall) instead of forcing it into support queues.



\## Limitations

\- Golden labels: AI-prelabeled, human-audited by a single annotator → subjectivity,

&#x20; especially for `other` (\~10 ambiguous rows identified)

\- Thresholds tuned on the eval set; a held-out set would give unbiased estimates

\- n = 200; per-class metrics carry wide confidence intervals

\- Escalation decisions not yet evaluated (future work)



\## Future work

\- Larger golden set + second annotator for inter-annotator agreement

\- Escalation-route evaluation

\- Taxonomy revision: several tweets (packaging, courier conduct, UI feedback)

&#x20; fit no category cleanly — taxonomy coverage, not classifier skill, is the ceiling



