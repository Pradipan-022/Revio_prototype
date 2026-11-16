# Revio AI Triage: Aspect Attribution & Risk-Aware Reply Routing

A self-contained Streamlit proof of concept for Cynbit Technologies' Revio
reputation management platform. It uses a Hugging Face zero-shot NLI model to
attribute review aspects and classify risk, then applies explicit three-lane
routing policy for a fictional multi-location QSR and café chain.

## Setup

Requires Python 3.10 or later.

```bash
cd revio-triage-poc
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first analysis downloads `valhalla/distilbart-mnli-12-1` from Hugging Face
and runs inference on CPU. Allow network access and enough disk space for the
model cache. The model is loaded lazily; if dependencies or model download are
unavailable, the UI reports the error rather than substituting a heuristic
classifier.

Run the policy verification script without downloading the model:

```bash
python test_engine.py
```

## Walkthrough

1. Open **Review inspector** and analyze the prefilled café review, or enter a
   branch, rating, and review of your own.
2. Observe the predicted aspect, risk category/confidence, rationale, and lane.
3. Open **Branch dashboard** and run the included synthetic multi-location
   review set, or upload a UTF-8 CSV.
4. Inspect the branch-level lane counts, triage table, and manager escalation
   details.

CSV columns: `review` is required; `branch`, `rating` (1–5), and `reviewer` are
optional. Blank review rows are ignored.

## Architecture

- `engine.py` — typed Pydantic request/result schemas, lazy Hugging Face
  zero-shot NLI adapter with descriptive aspect/risk hypotheses, deterministic
  high-risk cue overrides, confidence-gated auto-publish, reply drafting, and
  the routing policy. A small classifier protocol permits policy tests without
  model downloads.
- `app.py` — one Streamlit application with the review inspector and
  multi-branch dashboard. It does not publish externally or require an API
  server.
- `test_engine.py` — deterministic routing cases, model-independent latency
  measurement, and pass/fail accuracy output.

## Routing and role alignment

| Lane | Action | Revio role |
| --- | --- | --- |
| A · Auto-Publish | Routine praise / low-risk feedback; approved AI reply may publish immediately | Admin policy governs eligibility |
| B · Advisor Draft Queue | Specific branch-level operations issue; verify draft before publishing | Advisor |
| C · Manager Crisis Hold | Safety, hygiene, fraud, or legal/regulatory signal; public auto-reply blocked | Manager |

Admin policy owns the lane thresholds; it does not mean every Lane A item needs
an individual admin approval. Lane C has no reply draft. The included phrase
guard conservatively raises certain explicit safety, payment-fraud, and legal
signals to Lane C even when the model score is uncertain. Low-confidence
non-crisis predictions stay in the Advisor queue rather than auto-publishing.
This PoC is decision-support only and does not connect to review-provider APIs.
