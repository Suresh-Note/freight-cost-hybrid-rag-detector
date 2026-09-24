import json
from pathlib import Path

import pandas as pd

from src.anomaly_detector import detect_anomalies
from src.baseline import (
    calculate_own_history_baseline,
    calculate_similar_route_baseline,
)
from src.config import (
    FINAL_OUTPUT_FILE,
    GROQ_MODEL,
    LOGS_DIR,
    OUTPUT_COLUMNS,
    TOKEN_COST_LOG_FILE,
)
from src.data_loader import load_context_notes, load_shipments
from src.preprocessing import preprocess_shipments
from src.rag.grounded_generator import GroundedGenerator
from src.rag.rag_pipeline import RAGPipeline
from src.weekly_metrics import calculate_weekly_metrics


# ============================================================
# OUTPUT FORMATTING
# ============================================================


def format_percentage(
    value: float | None,
    description: str,
) -> str:
    """
    Format percentage comparison for the final CSV.
    """

    if pd.isna(value):
        return "N/A"

    sign = "+" if value >= 0 else ""

    return f"{sign}{value:.1f}% {description}"


def build_comparison_fields(row: pd.Series) -> tuple[str, str]:
    """
    Build the two human-readable comparison fields required
    by the final output format.
    """

    own_history = format_percentage(
        row["vs_own_history_pct"],
        "vs this route's past average",
    )

    similar_routes = format_percentage(
        row["vs_similar_routes_pct"],
        "vs similar-length routes this week",
    )

    return own_history, similar_routes


# ============================================================
# DETERMINISTIC REASONS
# ============================================================


def unexplained_reason_no_note() -> str:
    """
    Reason used when an anomaly has absolutely no retrieved context note.
    """
    return (
        "No matching note found for this route or date range. "
        "Cost rise looks unexplained and worth a human review."
    )


def unexplained_reason_rejected_note(note_id: str, note_date: str, note_text: str) -> str:
    """
    Reason used when a note was retrieved but rejected by the validator.
    """
    if "Context: " in note_text:
        actual_note = note_text.split("Context: ")[1]
    else:
        actual_note = note_text
        
    snippet = actual_note.split('.')[0][:60].strip()
    if len(actual_note.split('.')[0]) > 60:
        snippet += "..."
        
    return (
        f"The closest note ({note_id}, {note_date}) mentions \"{snippet}\" "
        "-- it does not describe a reason for a cost rise on this route. "
        "No genuine justification found; flagged for review."
    )


def normal_reason() -> str:
    """
    Reason for route-weeks that do not cross the anomaly threshold.
    """

    return (
        "No significant cost increase detected against the "
        "configured anomaly threshold."
    )


# ============================================================
# RETRIEVAL QUERY
# ============================================================


def build_retrieval_query(row: pd.Series) -> str:
    """
    Build a retrieval query using only information available
    from the route-week record.

    The query does not assume that any particular cause exists.
    """

    return (
        f"shipping cost increase for route {row['route']} "
        f"during week {row['week_of'].date()} "
        f"cost per tonne-km {row['cost_per_tonne_km']:.6f} "
        f"possible freight cost surcharge fuel toll weather "
        f"festival demand truck availability disruption"
    )


# ============================================================
# TOKEN / COST LOGGING
# ============================================================


def create_token_cost_log(
    total_input_tokens: int,
    total_output_tokens: int,
    total_tokens: int,
    llm_calls: int,
    model: str | None,
) -> dict:
    """
    Create the token and cost summary.

    Pricing below is for the selected Groq GPT-OSS 120B model.
    """

    # Groq published pricing for GPT-OSS 120B:
    # $0.15 / 1M input tokens
    # $0.60 / 1M output tokens

    input_price_per_million = 0.15
    output_price_per_million = 0.60

    estimated_input_cost = (
        total_input_tokens / 1_000_000
    ) * input_price_per_million

    estimated_output_cost = (
        total_output_tokens / 1_000_000
    ) * output_price_per_million

    estimated_total_cost = (
        estimated_input_cost + estimated_output_cost
    )

    return {
        "provider": "groq",
        "model": model,
        "llm_calls": llm_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "published_input_price_usd_per_1m_tokens": input_price_per_million,
        "published_output_price_usd_per_1m_tokens": output_price_per_million,
        "estimated_input_cost_usd": round(
            estimated_input_cost,
            8,
        ),
        "estimated_output_cost_usd": round(
            estimated_output_cost,
            8,
        ),
        "estimated_total_cost_usd": round(
            estimated_total_cost,
            8,
        ),
    }


def save_token_cost_log(log_data: dict) -> None:
    """
    Save token/cost information as JSON.
    """

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        TOKEN_COST_LOG_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            log_data,
            file,
            indent=2,
        )


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_pipeline() -> pd.DataFrame:
    """
    Run the complete Smart Shipping Cost Assistant pipeline.

    Pipeline:

    1. Load data
    2. Preprocess shipments
    3. Calculate weekly metrics
    4. Calculate own-history baseline
    5. Calculate similar-route baseline
    6. Detect candidate anomalies
    7. Initialize strong RAG pipeline
    8. Retrieve, rerank and validate evidence
    9. Generate grounded explanation only when justified
    10. Create final output
    11. Save output and token/cost log
    """

    print("=" * 70)
    print("FreightTiger Smart Shipping Cost Assistant")
    print("=" * 70)

    # ========================================================
    # STEP 1 — LOAD DATA
    # ========================================================

    print("\n[1/10] Loading data...")

    shipments = load_shipments()
    context_notes = load_context_notes()

    print(f"Shipments loaded: {len(shipments)}")
    print(f"Context notes loaded: {len(context_notes)}")

    # ========================================================
    # STEP 2 — PREPROCESS
    # ========================================================

    print("\n[2/10] Preprocessing shipments...")

    shipments = preprocess_shipments(
        shipments
    )

    print(f"Preprocessed rows: {len(shipments)}")
    print(
        f"Unique routes: {shipments['route'].nunique()}"
    )
    print(
        f"Unique weeks: {shipments['week_of'].nunique()}"
    )

    # ========================================================
    # STEP 3 — WEEKLY METRICS
    # ========================================================

    print("\n[3/10] Calculating weekly metrics...")

    weekly = calculate_weekly_metrics(
        shipments
    )

    print(f"Weekly route rows: {len(weekly)}")

    # ========================================================
    # STEP 4 — OWN HISTORY BASELINE
    # ========================================================

    print("\n[4/10] Calculating own-history baseline...")

    weekly = calculate_own_history_baseline(
        weekly
    )

    # ========================================================
    # STEP 5 — SIMILAR ROUTE BASELINE
    # ========================================================

    print("\n[5/10] Calculating similar-route baseline...")

    weekly = calculate_similar_route_baseline(
        weekly
    )

    # ========================================================
    # STEP 6 — ANOMALY DETECTION
    # ========================================================

    print("\n[6/10] Detecting candidate anomalies...")

    weekly = detect_anomalies(
        weekly
    )

    candidate_count = int(
        weekly["is_candidate_anomaly"].sum()
    )

    print(
        f"Candidate anomalies: {candidate_count}"
    )

    # ========================================================
    # STEP 7 — STRONG RAG + GROUNDED GENERATOR
    # ========================================================

    print("\n[7/10] Initializing strong RAG pipeline...")

    rag_pipeline = RAGPipeline()

    print("\nInitializing grounded LLM generator...")

    grounded_generator = GroundedGenerator()

    print(
        "Strong RAG ready: "
        "Dense + BM25 + Hybrid + Cross-Encoder + Validator"
    )

    # ========================================================
    # PROCESS ROUTE-WEEK RECORDS
    # ========================================================

    print("\n[8/10] Validating anomalies and evidence...")

    output_rows = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    llm_calls = 0
    model_used = GROQ_MODEL

    justified_count = 0
    unexplained_count = 0

    # Stable ordering is important for reproducibility.
    weekly = weekly.sort_values(
        ["route", "week_of"]
    ).reset_index(drop=True)

    for _, row in weekly.iterrows():

        own_history_text, similar_routes_text = (
            build_comparison_fields(row)
        )

        # ----------------------------------------------------
        # NON-CANDIDATE
        # ----------------------------------------------------

        if not row["is_candidate_anomaly"]:

            output_rows.append(
                {
                    "route": row["route"],
                    "week_of": row["week_of"],
                    "cost_per_tonne_km": row[
                        "cost_per_tonne_km"
                    ],
                    "vs_own_history": own_history_text,
                    "vs_similar_routes": similar_routes_text,
                    "flagged": "No",
                    "matched_note_id": "",
                    "reason": normal_reason(),
                }
            )

            continue

        # ----------------------------------------------------
        # CANDIDATE ANOMALY
        # ----------------------------------------------------

        query = build_retrieval_query(row)

        week_of_string = row["week_of"].strftime(
            "%Y-%m-%d"
        )

        # ----------------------------------------------------
        # STRONG RAG:
        # Dense + BM25
        # Hybrid retrieval
        # Cross-encoder reranking
        # Route/date validation
        # Strict evidence validation
        # ----------------------------------------------------

        retrieved_evidence = rag_pipeline.retrieve_evidence(
            query=query,
            route=row["route"],
            week_of=week_of_string,
            top_k=5,
            candidate_k=10,
        )
        
        valid_evidence = next((item for item in retrieved_evidence if item.evidence_valid), None)

        # ----------------------------------------------------
        # VALID SUPPORTING NOTE
        # ----------------------------------------------------

        if valid_evidence is not None:

            justified_count += 1

            note_id = valid_evidence.note_id
            note_date = valid_evidence.note_date

            note_text = valid_evidence.note_text

            explanation = grounded_generator.generate(
                route=row["route"],
                week_of=week_of_string,
                cost_per_tonne_km=float(
                    row["cost_per_tonne_km"]
                ),
                vs_own_history=(
                    None
                    if pd.isna(
                        row["vs_own_history_pct"]
                    )
                    else float(
                        row["vs_own_history_pct"]
                    )
                ),
                vs_similar_routes=(
                    None
                    if pd.isna(
                        row["vs_similar_routes_pct"]
                    )
                    else float(
                        row["vs_similar_routes_pct"]
                    )
                ),
                note_id=note_id,
                note_date=note_date,
                note_text=note_text,
            )

            total_input_tokens += (
                explanation.input_tokens
            )

            total_output_tokens += (
                explanation.output_tokens
            )

            total_tokens += (
                explanation.total_tokens
            )

            llm_calls += 1

            output_rows.append(
                {
                    "route": row["route"],
                    "week_of": row["week_of"],
                    "cost_per_tonne_km": row[
                        "cost_per_tonne_km"
                    ],
                    "vs_own_history": own_history_text,
                    "vs_similar_routes": similar_routes_text,
                    "flagged": "No (justified)",
                    "matched_note_id": note_id,
                    "reason": explanation.reason,
                }
            )

        # ----------------------------------------------------
        # NO VALID SUPPORTING NOTE
        # ----------------------------------------------------

        else:

            unexplained_count += 1

            if len(retrieved_evidence) > 0:
                closest = retrieved_evidence[0]
                reason_text = unexplained_reason_rejected_note(
                    closest.note_id,
                    closest.note_date,
                    closest.note_text
                )
            else:
                reason_text = unexplained_reason_no_note()

            output_rows.append(
                {
                    "route": row["route"],
                    "week_of": row["week_of"],
                    "cost_per_tonne_km": row[
                        "cost_per_tonne_km"
                    ],
                    "vs_own_history": own_history_text,
                    "vs_similar_routes": similar_routes_text,
                    "flagged": "Yes",
                    "matched_note_id": "",
                    "reason": reason_text,
                }
            )

    # ========================================================
    # STEP 9 — BUILD FINAL OUTPUT
    # ========================================================

    print("\n[9/10] Building final output...")

    result = pd.DataFrame(
        output_rows
    )

    # Ensure exact column order.
    result = result[OUTPUT_COLUMNS]

    # Final CSV formatting.
    result["week_of"] = pd.to_datetime(
        result["week_of"]
    ).dt.strftime("%Y-%m-%d")

    result["cost_per_tonne_km"] = (
        result["cost_per_tonne_km"]
        .round(2)
    )

    # Ensure blank note IDs are actually blank.
    result["matched_note_id"] = (
        result["matched_note_id"]
        .fillna("")
        .astype(str)
    )

    # ========================================================
    # STEP 10 — SAVE OUTPUT + TOKEN LOG
    # ========================================================

    print("\n[10/10] Saving output and logs...")

    Path(FINAL_OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        FINAL_OUTPUT_FILE,
        index=False,
    )

    token_log = create_token_cost_log(
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        llm_calls=llm_calls,
        model=model_used,
    )

    save_token_cost_log(
        token_log
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Total route-weeks: {len(result)}"
    )

    print(
        f"Candidate anomalies: {candidate_count}"
    )

    print(
        f"Justified anomalies: {justified_count}"
    )

    print(
        f"Unexplained anomalies: {unexplained_count}"
    )

    print(
        f"LLM calls: {llm_calls}"
    )

    print(
        f"Input tokens: {total_input_tokens}"
    )

    print(
        f"Output tokens: {total_output_tokens}"
    )

    print(
        f"Total tokens: {total_tokens}"
    )

    print(
        f"Output file: {FINAL_OUTPUT_FILE}"
    )

    print(
        f"Token log: {TOKEN_COST_LOG_FILE}"
    )

    print("=" * 70)

    return result


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================


if __name__ == "__main__":
    run_pipeline()
