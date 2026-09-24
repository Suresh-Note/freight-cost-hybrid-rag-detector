# FreightTiger Smart Shipping Cost Assistant
## 10-Minute Walkthrough & Interview Cheat Sheet

---

## 📋 Quick Project Summary

**What it does:** Monitors freight shipping costs across 7 routes, detects unjustified cost spikes using statistical baselines, retrieves supporting context notes using a Hybrid RAG pipeline, and explains them in plain English using a grounded LLM — without hallucinating.

**Results:**
- 728 total route-weeks processed
- 17 unexplained anomalies flagged for review
- 3 anomalies justified with cited evidence (N001, N002)
- 9/9 evaluation harness PASS (100% accuracy)
- Only 3 LLM calls made — total cost: ~$0.0004 USD

---

## 📁 Directory Structure

```
smart_assistances/
├── data/                          # Input data
│   ├── shipment_records.csv       # Raw shipment records (2940 rows)
│   ├── context_notes.csv          # 10 context notes (RAG knowledge base)
│   └── sample_output_format_v2.csv
├── src/                           # Core pipeline
│   ├── config.py                  # All constants (threshold, model, paths)
│   ├── data_loader.py             # CSV loading + schema validation
│   ├── preprocessing.py           # Route + Monday week_of creation
│   ├── weekly_metrics.py          # Cost per tonne-km aggregation
│   ├── baseline.py                # Own-history + similar-route baselines
│   ├── anomaly_detector.py        # 20% threshold flagging
│   ├── evidence_validator.py      # Deterministic regex validation (anti-hallucination)
│   ├── pipeline.py                # Main 10-step orchestrator
│   └── rag/                       # Advanced RAG system
│       ├── document_builder.py    # Note → searchable document
│       ├── vector_store.py        # Dense retrieval (SentenceTransformer)
│       ├── bm25_retriever.py      # Sparse retrieval (BM25)
│       ├── hybrid_retriever.py    # 60% dense + 40% BM25 fusion
│       ├── reranker.py            # Cross-encoder reranking
│       ├── grounded_generator.py  # Groq LLM @ Temperature 0.0
│       └── rag_pipeline.py        # End-to-end RAG orchestrator
├── eval/                          # Evaluation & testing
│   ├── labeled_set.csv            # 9 hand-labeled edge cases
│   ├── run_evaluation.py          # Automated eval harness (9/9 PASS)
│   ├── run_reproducibility.py     # 3-run determinism test
│   └── evaluation_results.json    # Stored results
├── output/
│   └── final_output.csv           # Main pipeline output (728 rows)
├── logs/
│   ├── token_cost_log.json        # Token usage + estimated cost
│   └── reproducibility_report.txt # 3-run PASS report
├── run.py                         # Entry point
├── app.py                         # Streamlit dashboard
└── requirements.txt
```

---

## 🏗️ System Architecture (How to Explain It)

```
shipment_records.csv
        │
        ▼
[1] Preprocess: create route + Monday week_of
        │
        ▼
[2] Weekly Metrics: cost_per_tonne_km = total_freight_cost / (qty_tonnes × distance_km)
        │
        ├──▶ [3a] Own-History Baseline: 8-week trailing average (shift(1), NO look-ahead)
        │
        └──▶ [3b] Similar-Route Baseline: same week, same route_type, exclude self
                │
                ▼
        [4] Anomaly Detection: flag if either baseline exceeded by ≥ 20%
                │
                ├── NOT anomaly → "No" → Normal reason (no LLM call)
                │
                └── IS anomaly → Run HYBRID RAG pipeline:
                        │
                        ├── Dense Retrieval (all-MiniLM-L6-v2)
                        ├── BM25 Sparse Retrieval (rank-bm25)
                        ├── Hybrid Fusion (60% dense + 40% BM25)
                        ├── Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
                        └── Evidence Validator (deterministic regex)
                                │
                                ├── VALID note → Groq LLM generates explanation → "No (justified)"
                                └── INVALID/NONE → Deterministic fallback → "Yes" (flagged)
```

---

## 🎤 Expected Interview Questions & Model Answers

### Q1: Walk me through your system architecture.
> "I built a 10-step pipeline. It starts by loading the raw shipments, calculating cost per tonne-km weekly, then computing two baselines: an 8-week own-history trailing average with `shift(1)` to prevent look-ahead, and a same-week peer average across similar route types. Any route-week exceeding either baseline by 20% becomes a candidate anomaly. For each candidate, I run a 4-stage Hybrid RAG system that fuses dense semantic vectors with BM25 keyword search, reranks using a Cross-Encoder, and then validates the retrieved note deterministically with regex patterns. Only validated notes trigger an LLM call to generate the plain-English explanation."

### Q2: Why 20% threshold?
> "The brief didn't specify a threshold, so I chose 20% as a practical engineering heuristic. It's high enough to ignore normal weekly fluctuations (±3-5%), but low enough to catch genuine spikes like the Ahmedabad-Mumbai festival surcharge at +29.5%."

### Q3: How did you prevent hallucination?
> "I separated the verdict from the explanation. The verdict—justified vs. unexplained—is decided by a deterministic Python regex validator (`evidence_validator.py`), not the LLM. The validator looks for explicit positive phrases like 'higher trip costs' and 'surcharge applied', and explicitly rejects notes containing negative phrases like 'costs were not affected' or 'without a rate change'. The LLM at temperature 0.0 only generates the final plain-English sentence after the validator has already confirmed validity."

### Q4: How do you know your system is right?
> "I built a lightweight evaluation harness. I hand-labeled 9 edge cases covering all the tricky scenarios mentioned in the brief: valid notes, wrong-date notes, wrong-route notes, and notes with negative phrasing. My system scores 9/9 (100%). I also ran a 3-run reproducibility check that proves all deterministic fields are identical across runs."

### Q5: Why Hybrid RAG instead of just vectors?
> "Dense vectors are great for semantic similarity but weak at exact keyword matching. BM25 is excellent at matching route names like 'Ahmedabad-Mumbai' but misses semantic relationships. By combining both (60% dense, 40% sparse) and reranking with a Cross-Encoder, I get the best of both worlds. In a real FreightTiger environment with thousands of notes, this approach scales much better than either alone."

### Q6: Are you doing look-ahead in your baseline?
> "Absolutely not. In `baseline.py`, I use pandas `.shift(1)` before calculating the rolling window. This ensures that the baseline for any given week uses only data strictly prior to that week's Monday — exactly how a real-time monitoring system would work."

### Q7: Why only 3 LLM calls for 728 rows?
> "Because I only trigger the LLM for the 3 anomalies that pass the evidence validator. All other rows — including the 17 unexplained anomalies — use deterministic fallback text. This keeps the total cost at ~$0.0004 USD, which the brief specifically asks to track and minimize."

### Q8: Why not use LangChain?
> "LangChain is excellent for rapid prototyping. But for this use case — monitoring financial freight costs where hallucination could cause incorrect business decisions — I needed 100% control over every step. Building the pipeline natively means I can write unit tests for every individual component, guarantee deterministic outputs, and explain exactly what every line does. That's much harder to achieve when you surrender control to a heavy framework's abstractions."

---

## ✅ Pre-Interview Checklist (Do This at 10:45 AM)

1. Run `pytest tests/ -v` → confirm all tests pass
2. Run `python run.py` → confirm pipeline completes in ~30-60 seconds
3. Run `streamlit run app.py` → confirm dashboard opens in browser
4. Have `output/final_output.csv` open in a text editor to show the date format
5. Have `src/evidence_validator.py` open in your IDE (most impressive file to show)
6. Have `eval/evaluation_results.json` open to quickly show 9/9 PASS
7. Have `logs/token_cost_log.json` open to show responsible LLM usage

---

## 🧪 Test Questions for the Streamlit AI Assistant

Type these into the "Ask your question" box during the demo:

1. *"Was there a surcharge or any truck availability issues on the Ahmedabad-Mumbai route in January 2025?"*
   → Shows: RAG correctly retrieves N002, references festival surcharge

2. *"Did the highway maintenance on the Mumbai-Delhi route cause a cost increase?"*
   → Shows: AI correctly says NO — it reads N005 and reports costs were NOT affected (anti-hallucination demo)

3. *"Were any shipping routes disrupted by bad weather or natural disasters?"*
   → Shows: Dense vector search understands "natural disasters" = N001's "heavy flooding"

4. *"Did fuel prices have an impact on transportation costs this year?"*
   → Shows: Global "All Routes" notes are correctly handled (N003)

5. *"What impact did the new vehicle tracking mandate have on our freight rates?"*
   → Shows: AI correctly reports NO cost impact (N010 explicitly says costs absorbed without rate change)
