AI Customer Support Agent — Amazon
An AI customer-support agent built using the Customer Support on Twitter (TWCS) dataset.

The system is designed around three tasks:

Intent classification — classify an incoming customer message into a small Amazon-specific intent taxonomy.
Reply drafting — retrieve historically similar Amazon support replies and use them as grounding for a response.
Routing — decide whether the message should be auto_handle or escalate, with a stated reason.
The project also includes baselines, a golden evaluation set, error analysis, and an LLM-as-judge workflow.

Important: Generated evaluation artifacts and the raw dataset are not necessarily committed to GitHub. The commands below explain how to reproduce them locally.

1. Project Structure
.
├── agent_pipeline.py
├── analyze_errors.py
├── apply_ai_labels.py
├── audit_brands.py
├── baselines.py
├── build_amazon_dataset.py
├── build_retrieval_index.py
├── convert_review_to_jsonl.py
├── derive_taxonomy.py
├── evaluate_agent.py
├── human_agreement.py
├── llm_judge.py
├── make_golden_set.py
├── make_review_sheet.py
├── prelabel_golden.py
├── rescore_golden.py
├── search_demo.py
├── REPORT.md
├── README.md
├── requirements.txt
├── sample.csv
├── twcs.csv
└── results/
Important scripts
File	Purpose
audit_brands.py	Inspect brands available in the dataset
build_amazon_dataset.py	Create the Amazon-specific dataset
derive_taxonomy.py	Derive the Amazon intent taxonomy
build_retrieval_index.py	Build the semantic retrieval index
agent_pipeline.py	Run classification, retrieval, reply drafting and routing
make_golden_set.py	Create the evaluation set
evaluate_agent.py	Evaluate intent classification
baselines.py	Run majority and TF-IDF/Logistic Regression baselines
analyze_errors.py	Analyze incorrect predictions
llm_judge.py	Evaluate drafted replies using an LLM
human_agreement.py	Compare human evaluation with LLM-judge scores
2. Environment
Recommended:

Python 3.10+
Windows, macOS or Linux
Internet connection for installing packages and downloading models
Create a virtual environment.

Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
Install dependencies:

pip install -r requirements.txt
3. Dataset
This project uses the Customer Support on Twitter (TWCS) dataset.

Dataset:

thoughtvector/customer-support-on-twitter
The raw dataset is intentionally not committed to GitHub because of its size.

Place the dataset CSV in the project according to the expectations of the project scripts.

If twcs.csv already exists in the project directory, it can be used directly.

4. Running the Pipeline
Run the following steps in order.

Step 1 — Audit brands
python audit_brands.py
This inspects the brands present in the dataset and allows Amazon to be selected.

Step 2 — Build Amazon dataset
python build_amazon_dataset.py
This extracts the Amazon-related support conversations.

Step 3 — Derive the taxonomy
python derive_taxonomy.py
The current Amazon taxonomy contains:

delivery_issue
damaged_item_or_product_issue
prime_membership_issue
support_process_complaint
other
The taxonomy was derived from the Amazon support data rather than importing a generic customer-support taxonomy.

Step 4 — Build retrieval index
python build_retrieval_index.py
This creates the semantic retrieval artifacts used to find historically similar Amazon support interactions.

Step 5 — Create the golden evaluation set
python make_golden_set.py
The current evaluation uses:

200 examples
This satisfies the assignment requirement of 150–250 hand-labelled examples.

Step 6 — Run the agent
python agent_pipeline.py
The agent performs:

Customer message
       ↓
Intent classification
       ↓
Historical reply retrieval
       ↓
Draft reply
       ↓
AUTO-HANDLE / ESCALATE
The generated output is written to:

results/agent_outputs.jsonl
Each output contains information such as:

customer_text
predicted_intent
intent_confidence
all_similarities
route
route_reason
drafted_reply
grounding
5. Intent Evaluation
Run:

python evaluate_agent.py
This reports:

Precision
Recall
F1
Intent accuracy
Prediction mismatches
Current evaluation:

Golden set size: 200
Intent accuracy: 0.51
Per-class results currently include:

Intent	Precision	Recall	F1
delivery_issue	0.60	0.50	0.55
damaged_item_or_product_issue	0.17	0.32	0.22
prime_membership_issue	0.70	0.74	0.72
support_process_complaint	0.53	0.58	0.56
other	0.33	0.14	0.20
6. Baselines
Run:

python baselines.py
The project compares the agent against two baselines.

Baseline 1 — Majority Classifier
Always predicts the most common intent.

Baseline 2 — TF-IDF + Logistic Regression
A conventional text classification baseline using TF-IDF features and Logistic Regression with 5-fold cross-validation.

Current results
System	Accuracy	Macro-F1
Majority baseline	0.30	0.09
TF-IDF + Logistic Regression	0.30	0.18
Agent	0.51	0.45
The agent therefore currently outperforms both baselines on the recorded golden-set evaluation.

7. LLM Reply Evaluation
The project includes an LLM-as-judge component in:

llm_judge.py
The judge evaluates drafted replies on:

Relevance
Helpfulness
Groundedness
Brand consistency
Hallucination avoidance
Overall quality
The judge should evaluate whether the response actually addresses the customer rather than rewarding generic polite language.

OpenRouter
The LLM judge can use an OpenRouter model.

Example PowerShell configuration:

$env:OPENROUTER_API_KEY="YOUR_API_KEY"
$env:OPENROUTER_MODEL="nvidia/nemotron-3-super-120b-a12b:free"
Then run:

python llm_judge.py
Results are saved to:

results/judge_results.json
Security
Never commit the API key to GitHub.

Do not place the key directly inside Python source code.

If an API key is accidentally exposed, revoke it and create a new one.

8. Human Evaluation
Human evaluation must use genuine human ratings.

AI-generated ratings must not be presented as human evaluation.

The review uses a 1–5 scale:

Score	Meaning
1	Completely wrong or unrelated
2	Mostly wrong
3	Partially useful
4	Good
5	Excellent and directly helpful
The reviewer should judge the actual drafted reply against the customer's message.

After completing the human review, run:

python human_agreement.py
This is used to compare human ratings with the LLM judge.

9. Error Analysis
Run:

python analyze_errors.py
The error analysis examines real examples of incorrect or poor behavior.

Important observed failure patterns include:

1. Unrelated replies
Some delivery complaints receive replies that discuss unrelated account or packaging issues.

2. Low-confidence escalation
Some legitimate product or service questions are escalated simply because the classifier is uncertain.

3. Difficult other category
Many messages do not fit cleanly into the small four-domain taxonomy and therefore create confusion around other.

4. Keyword/rule interference
Simple keyword rules can cause the system to select an incorrect intent when a keyword appears in an unrelated context.

5. Generic support responses
A response may sound professional but still fail to answer the customer's actual question.

These examples are important because the headline accuracy number alone does not reveal these failure modes.

10. Routing
The agent makes one of two routing decisions:

AUTO-HANDLE
or:

ESCALATE
The system currently considers factors such as:

legal threats
refund/chargeback requests
account security concerns
urgent/time-bound situations
low intent confidence
The output contains a routing reason:

route
route_reason
The routing decision is intentionally conservative for higher-risk situations.

11. What the System Does NOT Build
This project is an evaluation prototype.

It does not:

access real Amazon customer accounts
modify orders
issue real refunds
access payment information
guarantee delivery dates
contact customers
replace human support agents
provide a general-purpose support taxonomy
assume historical replies are always correct
claim that retrieval alone guarantees a correct answer
12. Reproducibility
A basic local reproduction is:

pip install -r requirements.txt

python audit_brands.py
python build_amazon_dataset.py
python derive_taxonomy.py
python build_retrieval_index.py
python make_golden_set.py
python agent_pipeline.py
python evaluate_agent.py
python baselines.py
python analyze_errors.py
For LLM reply evaluation:

python llm_judge.py
For human/judge agreement:

python human_agreement.py
The first execution may take longer because the sentence-transformer model needs to be downloaded.

13. Generated Results
The following files are generated during the workflow:

results/
├── agent_outputs.jsonl
├── brand_audit.json
├── decision_log.md
├── draft_taxonomy.json
├── golden_set_ai_labels.jsonl
├── golden_set_labeled.jsonl
├── golden_set_prelabel.jsonl
├── golden_set_unlabeled.jsonl
├── judge_results.json
├── my_labels.json
└── review_sheet.csv
These files contain evaluation artifacts and generated outputs.

Large datasets and model/index artifacts should generally remain outside GitHub.

14. Current Headline Result
On the current 200-example golden evaluation:

Intent accuracy = 0.51
Macro-F1 = 0.45
The agent outperforms the two implemented baselines:

Majority:
Accuracy = 0.30
Macro-F1 = 0.09

TF-IDF + Logistic Regression:
Accuracy = 0.30
Macro-F1 = 0.18

Agent:
Accuracy = 0.51
Macro-F1 = 0.45
However, 0.51 accuracy should not be interpreted as “51% of customer support conversations are solved correctly.”

Intent classification is only one part of the system. Reply relevance, grounding, hallucination risk, and escalation quality must also be considered.

15. Limitations
The current evaluation has several limitations:

The golden set contains 200 examples.
Human evaluation requires genuine human annotation.
The taxonomy contains a difficult other class.
Some thresholds were tuned during development.
Intent classification and escalation are separate problems.
Historical Twitter support conversations can be noisy or incomplete.
A retrieved historical reply can still be inappropriate for a new customer.
The current headline accuracy does not measure whether the customer actually received a successful resolution.
Routing quality needs its own evaluation rather than being inferred from intent accuracy.
These limitations should be considered when interpreting the results.

16. Assignment Deliverables
Assignment requirement	Project artifact
Runnable repository	Repository + README
150–250 evaluation examples	Golden-set artifacts
Automated metrics	evaluate_agent.py
Trivial baseline	Majority baseline in baselines.py
Simple baseline	TF-IDF + Logistic Regression in baselines.py
LLM-as-judge	llm_judge.py
Human agreement	human_agreement.py
Failure analysis	analyze_errors.py + REPORT.md
Decision log	results/decision_log.md
Report	REPORT.md
17. Future Work
With another week of development, the highest-value improvements would be:

Improve the taxonomy using additional human review.
Add a dedicated escalation gold set.
Improve retrieval so the selected historical reply is more closely matched to the customer's issue.
Add stronger safeguards against unrelated replies.
Use a second human annotator and measure inter-annotator agreement.
Evaluate reply quality independently from intent classification.
Test the system on a larger held-out dataset.
Separate development/tuning data from the final test set.
The most important next goal is not simply increasing the headline accuracy, but improving end-to-end support quality and safe routing.
