# 🚚 FreightTiger Smart Shipping Cost Assistant

> **24-Hour Case Study Submission — Software Engineering Intern (AI)**  
> A production-grade intelligent assistant that monitors weekly freight costs, detects unjustified cost spikes, and explains them in plain English using a Hybrid RAG pipeline — with zero hallucination.

---

## 📋 Table of Contents

1. [What It Does](#what-it-does)
2. [Quick Start](#quick-start)
3. [How the Pipeline Works](#how-the-pipeline-works)
4. [RAG Architecture](#rag-architecture)
5. [Anti-Hallucination Guardrails](#anti-hallucination-guardrails)
6. [Output Format](#output-format)
7. [How I Know It's Right](#how-i-know-its-right)
8. [Token & Cost Report](#token--cost-report)
9. [Reproducibility](#reproducibility)
10. [Project Structure](#project-structure)
11. [Design Decisions](#design-decisions)

---

## What It Does

Companies pay transporters to move goods. Over time, the cost per tonne-km on certain routes can quietly creep up — sometimes for a real reason (fuel prices, a toll change, monsoon disruption), sometimes for no reason at all.

This assistant:
1. **Monitors** weekly shipping cost per tonne-km across all routes
2. **Detects** weeks where the cost is suspiciously higher than normal
3. **Searches** a knowledge base of real-world context notes using a Hybrid RAG pipeline
4. **Validates** the retrieved note with deterministic rules — no hallucination possible
5. **Explains** justified anomalies in plain English, citing the exact note
6. **Flags** unexplained anomalies for human review

### The Output — Three Possible Verdicts Per Route-Week

| Verdict | Meaning | Count |
|---|---|---|
| `No` | Cost is normal — within expected range | 708 rows |
| `No (justified)` | Cost spiked, but the AI found a valid reason (e.g. festival surcharge) | 3 rows |
| `Yes` | Cost spiked with NO valid explanation — **needs human review** | 17 rows |

---

## Quick Start

```bash
# 1. Clone and set up environment
pip install -r requirements.txt

# 2. Add your Groq API key to .env
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Run the full pipeline (generates output/final_output.csv)
python run.py

# 4. Launch the interactive Streamlit dashboard
streamlit run app.py

# 5. Run the evaluation harness (proves it works correctly)
python -m eval.run_evaluation

# 6. Run the 3-run reproducibility check
python -m eval.run_reproducibility
```

---

## How the Pipeline Works

The system runs a **10-step pipeline** orchestrated by `src/pipeline.py`:

```
shipment_records.csv  ──────────────────────────────────────────┐
context_notes.csv     ──────────────────────────────────────────┤
                                                                 ▼
Step 1: Load & Validate Data
        → Schema check: required columns, no nulls, positive values

Step 2: Preprocess Shipments
        → Create route = "Origin-Destination"
        → Assign week_of = that Monday (Monday-to-Sunday weeks)

Step 3: Calculate Weekly Cost per Tonne-km
        → cost_per_tonne_km = total_freight_cost / (qty_tonnes × distance_km)
        → Aggregated per route per week

Step 4: Own-History Baseline
        → Trailing 8-week rolling average of cost_per_tonne_km
        → Uses .shift(1) — STRICTLY prior weeks only (no look-ahead)
        → Fewer than 8 prior weeks? Uses all available weeks

Step 5: Similar-Route Baseline
        → Average cost of all other routes with the same route_type
        → In the SAME week, excluding the route itself (no self-bias)

Step 6: Anomaly Detection
        → Flag if cost exceeds EITHER baseline by ≥ 20%

Step 7: Hybrid RAG Pipeline
        → For each flagged anomaly, search context_notes.csv
        → 4-stage retrieval (see RAG Architecture below)

Step 8: Evidence Validation
        → Deterministic regex check — NOT an LLM decision
        → Requires positive cost-impact language
        → Rejects notes with negative phrasing ("not affected")

Step 9: Grounded Explanation (LLM)
        → Only called for VALIDATED notes (3 out of 728 rows)
        → Temperature = 0.0 for deterministic output
        → Strictly grounded — cites exact note ID, route, week

Step 10: Save Output
        → output/final_output.csv  (728 rows, 8 columns)
        → logs/token_cost_log.json (token usage + cost)
```

---

## RAG Architecture

For each candidate anomaly, the system runs a **4-stage Hybrid RAG pipeline** to find the most relevant context note:

```
Query (route + week + cost data)
        │
        ├──► Stage 1: Dense Retrieval
        │    Model: all-MiniLM-L6-v2 (SentenceTransformer)
        │    Captures SEMANTIC similarity
        │    e.g. "surcharge" ≈ "cost increase" ≈ "higher rates"
        │
        ├──► Stage 2: BM25 Sparse Retrieval
        │    Library: rank-bm25
        │    Captures EXACT keyword matches
        │    e.g. "Ahmedabad-Mumbai", note IDs
        │
        ▼
Stage 3: Hybrid Fusion
        60% Dense score + 40% BM25 score (min-max normalised)
        Best of semantic + keyword worlds
        │
        ▼
Stage 4: Cross-Encoder Reranking
        Model: ms-marco-MiniLM-L-6-v2
        Jointly scores [Query + Document] together
        Most precise ranking — catches subtle relevance
        │
        ▼
Stage 5: Evidence Validation (Deterministic)
        Regex-based — NOT the LLM
        ✓ Route must match or note must apply to "All Routes"
        ✓ Note date must be within ±7 days of anomaly week
        ✓ Note must contain explicit positive cost-impact phrases
        ✗ Note must NOT contain negative evidence phrases
        │
        ├── VALID ──► LLM generates plain-English explanation
        └── INVALID ──► Python fallback — "No genuine justification found"
```

### Why Hybrid? Why Not Just Dense Vectors?

| Method | Strength | Weakness |
|---|---|---|
| Dense (vectors) | Finds "festival surcharge" when you search "cost increase" | Bad at exact route name matching |
| BM25 (keyword) | Excellent at exact terms like "Ahmedabad-Mumbai" | Misses semantic relationships |
| **Hybrid (ours)** | **Gets both — best recall and precision** | None |

---

## Anti-Hallucination Guardrails

This is the most important engineering decision in the entire system.

### The Problem With Naive RAG

A naive approach would ask the LLM: *"Does this note explain a cost increase? Yes or No?"*  
**This is dangerous.** LLMs can:
- Misread negative phrasing ("costs were NOT affected" → LLM says "yes, costs affected")
- Hallucinate reasons not present in the note
- Give different answers at different temperatures

### Our Solution: Deterministic Validation Before LLM

The **verdict** (justified vs. unexplained) is decided by `src/evidence_validator.py` — pure Python regex, never an LLM.

**Positive patterns** (note MUST contain at least one):
```
"higher trip costs", "surcharge applied", "pushing up transportation costs",
"higher costs", "increased freight rates", "cost increase", "forced longer detours"
```

**Negative patterns** (note must contain NONE of these):
```
"costs were not significantly affected", "without a rate change",
"no major disruptions", "stable demand", "returned to normal conditions",
"costs not affected", "freight movement remained normal"
```

**Result:** A note is valid ONLY IF it contains positive evidence AND no negative evidence. The LLM never gets to override this decision.

### Real Examples of Notes Correctly Rejected

| Note | Content | Decision |
|---|---|---|
| N005 | "highway maintenance caused minor delays; **costs were not significantly affected**" | ❌ REJECTED |
| N006 | "**stable demand** with **no major disruptions** reported" | ❌ REJECTED |
| N009 | "route **returned to normal conditions** after repairs" | ❌ REJECTED |
| N010 | "compliance costs absorbed **without a rate change**" | ❌ REJECTED |

---

## Output Format

The output exactly matches `data/sample_output_format_v2.csv`:

```
route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id,reason
```

### Sample Rows

**Justified anomaly (LLM-generated reason, note cited):**
```
Ahmedabad-Mumbai,2025-01-20,3.29,+29.5% vs this route's past average,
+22.5% vs similar-length routes this week,No (justified),N002,
"The cost rise on the Ahmedabad-Mumbai route in week 2025-01-20 (cost 3.29, 
+29.5% vs own history and +22.5% vs similar routes) is supported by Note N002, 
which states a regional festival week caused a temporary surcharge due to 
high demand and limited truck availability."
```

**Unexplained anomaly (deterministic fallback, blank note ID):**
```
Mumbai-Pune,2025-09-15,3.98,+9.2% vs this route's past average,
+23.6% vs similar-length routes this week,Yes,,
"The closest note (N002, 2025-01-20) mentions 'A regional festival week...' 
-- it does not describe a reason for a cost rise on this route. 
No genuine justification found; flagged for review."
```

**Normal row (no anomaly):**
```
Delhi-Jaipur,2024-01-01,3.10,-2.1% vs this route's past average,
-1.5% vs similar-length routes this week,No,,
"No significant cost increase detected against the configured anomaly threshold."
```

---

## How I Know It's Right

### 1. Labeled Evaluation Harness — 9/9 PASS (100%)

I hand-labeled 9 test cases covering all the tricky edge cases mentioned in the brief:

| Case | Tests | Expected | Result |
|------|-------|----------|--------|
| E001 | Valid festival surcharge note (N002) for Ahmedabad-Mumbai | justified | ✅ PASS |
| E002 | Valid flooding note (N001) for Chennai-Bangalore, Feb 24 | justified | ✅ PASS |
| E003 | Same flooding note still active one week later | justified | ✅ PASS |
| E004 | N009 says "returned to normal" — must be rejected | unexplained | ✅ PASS |
| E005 | N009 same rejection for week of Mar 10 | unexplained | ✅ PASS |
| E006 | N005 says "costs were not significantly affected" — reject | unexplained | ✅ PASS |
| E007 | N006 says "stable demand, no major disruptions" — reject | unexplained | ✅ PASS |
| E008 | N002 is for Ahmedabad-Mumbai, not Mumbai-Pune — wrong route | unexplained | ✅ PASS |
| E009 | No note at all — must say unexplained | unexplained | ✅ PASS |

**Run it yourself:**
```bash
python -m eval.run_evaluation
```

### 2. Unit Test Suite — 43/43 PASS

```bash
pytest tests/ -v
```

Tests cover:
- `test_anomaly_detector.py` — threshold logic, OR condition
- `test_baseline.py` — no look-ahead, shift(1), peer exclusion
- `test_data_loader.py` — schema validation, type checks
- `test_evidence_validator.py` — all positive and negative patterns
- `test_weekly_metrics.py` — formula correctness

### 3. Manual Spot-Checks

- ✅ Verified N001 (Chennai-Bangalore flooding) correctly justifies Feb 24 and Mar 3 anomalies
- ✅ Verified N002 (festival surcharge) correctly justifies Ahmedabad-Mumbai Jan 20 anomaly
- ✅ Verified that N004 ("affected routes are not part of this dataset") is correctly rejected
- ✅ Verified that notes with negative phrasing (N005, N006, N009, N010) never appear as justified

---

## Token & Cost Report

For one full run over the entire dataset (728 route-weeks, 20 candidate anomalies):

| Metric | Value |
|--------|-------|
| Provider | Groq |
| Model | openai/gpt-oss-120b |
| LLM Calls | **3** (only for validated anomalies) |
| Input Tokens | 1,196 |
| Output Tokens | 355 |
| Total Tokens | 1,551 |
| Estimated Cost | **~$0.0004 USD** |

**Why so few LLM calls?**  
The LLM is triggered ONLY when the deterministic evidence validator confirms a valid supporting note. All 17 unexplained anomalies and all 708 normal rows use zero LLM calls.

Full breakdown in `logs/token_cost_log.json`.

---

## Reproducibility

Three consecutive runs on identical input produce identical results for all deterministic fields:
`route`, `week_of`, `cost_per_tonne_km`, `vs_own_history`, `vs_similar_routes`, `flagged`, `matched_note_id`

Explanation wording may vary slightly because the LLM is generative, but the verdict never changes.

**LLM temperature is set to `0.0`** in `src/config.py` for maximum determinism.

```bash
python -m eval.run_reproducibility
# → logs/reproducibility_report.txt  (PASS)
```

---

## Project Structure

```
smart_assistances/
├── data/
│   ├── shipment_records.csv          # 2,940 raw shipment records
│   ├── context_notes.csv             # 10 context notes (RAG knowledge base)
│   └── sample_output_format_v2.csv   # Contract: exact column schema required
│
├── src/
│   ├── config.py                     # All constants (threshold, model, paths)
│   ├── data_loader.py                # CSV loading with strict schema validation
│   ├── preprocessing.py              # Route + Monday week_of creation
│   ├── weekly_metrics.py             # cost_per_tonne_km aggregation
│   ├── baseline.py                   # 8-week trailing + peer-route baselines
│   ├── anomaly_detector.py           # 20% threshold flagging (OR logic)
│   ├── evidence_validator.py         # Deterministic regex validation
│   ├── rag_assistant.py              # Streamlit Q&A assistant backend
│   └── pipeline.py                   # Main 10-step orchestrator
│
├── src/rag/
│   ├── document_builder.py           # Note → searchable document format
│   ├── vector_store.py               # Dense retrieval (SentenceTransformer)
│   ├── bm25_retriever.py             # Sparse retrieval (BM25)
│   ├── hybrid_retriever.py           # 60% Dense + 40% BM25 fusion
│   ├── reranker.py                   # Cross-encoder reranking
│   ├── grounded_generator.py         # Groq LLM wrapper (temp=0.0, 12 rules)
│   └── rag_pipeline.py               # End-to-end RAG orchestrator
│
├── eval/
│   ├── labeled_set.csv               # 9 hand-labeled edge cases
│   ├── run_evaluation.py             # Automated eval harness
│   ├── run_reproducibility.py        # 3-run determinism check
│   └── evaluation_results.json       # 9/9 PASS results
│
├── tests/                            # 43 unit tests (pytest)
│   ├── test_anomaly_detector.py
│   ├── test_baseline.py
│   ├── test_data_loader.py
│   ├── test_evidence_validator.py
│   └── test_weekly_metrics.py
│
├── output/
│   └── final_output.csv              # Main deliverable (728 rows)
│
├── logs/
│   ├── token_cost_log.json           # Token usage + estimated cost
│   └── reproducibility_report.txt    # 3-run PASS report
│
├── run.py                            # Entry point → runs full pipeline
├── app.py                            # Streamlit dashboard
├── requirements.txt
├── walkthrough.md                    # 10-min demo script + interview Q&A
└── .env                              # GROQ_API_KEY (not committed)
```

---

## Design Decisions

### Why 20% Anomaly Threshold?
The case study does not prescribe a specific threshold. I chose 20% because:
- It is high enough to ignore normal weekly fluctuations (typically ±3–7%)
- It is low enough to catch genuine spikes like Ahmedabad-Mumbai's +29.5% festival surge
- It uses OR logic — a spike against either baseline qualifies, making detection sensitive to any type of anomaly

### Why Hybrid RAG Instead of Pure Dense Vectors?
Dense embeddings understand semantic similarity ("surcharge" ≈ "higher rates") but struggle with exact keyword matching. BM25 is the opposite. By combining both (60%/40% weighted fusion) and reranking with a Cross-Encoder, we get maximum recall AND maximum precision. In a production system with thousands of notes, this approach scales far better than either method alone.

### Why Deterministic Validation Instead of LLM-as-Judge?
LLMs are non-deterministic and can misread negative phrasing. For a system that makes financial verdicts ("justified vs. unexplained"), determinism is non-negotiable. The regex validator runs before the LLM and cannot be overridden — guaranteeing that a note saying "costs were not affected" is always rejected, every single time.

### Why Only 3 LLM Calls?
The LLM is expensive, slow, and non-deterministic. I minimise its role to one specific task: writing a human-readable explanation for the 3 anomalies that already passed the deterministic validator. Everything else — detection, validation, fallback messages — is pure Python. This keeps the total cost at $0.0004 and makes the system highly scalable.

### Why No LangChain?
LangChain is excellent for rapid prototyping but introduces hidden abstractions that are hard to unit-test and debug. Building the pipeline natively gives 100% control over every component, allows comprehensive unit testing of each stage, and makes the system fully transparent and auditable — critical for enterprise trust.

---

## AI Assistant — Test Questions & Expected Results

The Streamlit dashboard includes an interactive AI Assistant tab (powered by the Hybrid RAG pipeline) that answers plain-English questions about freight costs, anomalies, and context notes. The following 10 questions were used to validate the assistant's behaviour:

### ✅ Group 1 — Positive Evidence (RAG Should Find a Justifying Note)

| # | Question | Expected Behaviour |
|---|---|---|
| Q1 | *"Was there a surcharge or truck availability issue on Ahmedabad-Mumbai in January 2025?"* | Retrieves N002, explains festival week caused +29.5% spike. Cites exact cost ₹3.29. |
| Q2 | *"Were any shipping routes disrupted by flooding or bad weather?"* | Retrieves N001 via semantic match ("bad weather" → "heavy flooding"), explains Chennai-Bangalore disruption Feb 24 – Mar 8. |
| Q3 | *"Did diesel or fuel prices rise and affect shipping costs?"* | Retrieves N003 (All Routes), explains nationwide diesel price rise pushed costs up 5–7%. |

### 🛡️ Group 2 — Negative Evidence (RAG Must Correctly Say NO — Anti-Hallucination)

| # | Question | Expected Behaviour |
|---|---|---|
| Q4 | *"Did the highway maintenance on Mumbai-Delhi cause a cost increase?"* | Retrieves N005, correctly answers NO — note explicitly states "costs were not significantly affected." |
| Q5 | *"Did the freight market see any disruptions in Q3 2025?"* | Retrieves N006 and N008, correctly answers NO — both notes confirm stable demand and normal freight movement. |
| Q6 | *"What impact did the new vehicle tracking mandate have on freight rates?"* | Retrieves N010, correctly answers NO rate change — compliance costs absorbed by transporters without changing rates. |

### 🔬 Group 3 — Deep Analytical Questions

| # | Question | Expected Behaviour |
|---|---|---|
| Q7 | *"Which route had the highest unexplained cost spikes?"* | RAG retrieves individual findings; for aggregate counts use the Dashboard tab filter. From data: Mumbai-Pune has 13 unexplained anomalies. |
| Q8 | *"Were there any cost anomalies in November or December 2025?"* | Retrieves Nov/Dec findings, correctly reports no anomalies crossed the 20% threshold in that period. |
| Q9 | *"Did road improvements on Delhi-Jaipur reduce shipping costs?"* | Retrieves N007 + multiple Delhi-Jaipur weekly findings, concludes road improvements did NOT reduce costs — later weeks actually showed cost increases flagged as anomalies. |
| Q10 | *"Were there any disruptions on the Kolkata-Bhubaneswar corridor?"* | Retrieves N008 with high confidence score (0.389), answers NO — freight movement remained normal. |

### Results Summary

All 10 questions were tested on the live system:

| Result | Count |
|---|---|
| Perfect retrieval + correct answer | 8 / 10 |
| Correct answer, partial retrieval (aggregate query limitation) | 1 / 10 (Q7) |
| Correct answer, evidence slightly outside target window | 1 / 10 (Q8) |

**Key validation:** Questions Q4, Q5, and Q6 are the most important — they prove the system correctly answers **NO** when context notes explicitly state costs were not affected. A hallucinating system would answer **YES** to all three. This system does not hallucinate.

### Running the AI Assistant

```bash
streamlit run app.py
# Navigate to the "💬 AI Assistant" tab
# Type any question about routes, anomalies, or context notes
```
