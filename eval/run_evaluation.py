import json
from pathlib import Path

import pandas as pd

from src.data_loader import load_context_notes
from src.evidence_validator import validate_note


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABELED_FILE = PROJECT_ROOT / "eval" / "labeled_set.csv"
RESULTS_FILE = PROJECT_ROOT / "eval" / "evaluation_results.json"


def run_evaluation():
    print("=" * 70)
    print("FreightTiger Evidence Validation Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load evaluation cases
    # --------------------------------------------------------

    labeled = pd.read_csv(LABELED_FILE)

    required_columns = [
        "case_id",
        "route",
        "week_of",
        "note_id",
        "expected_valid",
        "expected_verdict",
        "description",
    ]

    missing = [
        column
        for column in required_columns
        if column not in labeled.columns
    ]

    if missing:
        raise ValueError(
            f"Missing evaluation columns: {missing}"
        )

    # --------------------------------------------------------
    # Load context notes
    # --------------------------------------------------------

    notes = load_context_notes()

    results = []

    # --------------------------------------------------------
    # Evaluate every labeled case
    # --------------------------------------------------------

    for _, case in labeled.iterrows():

        case_id = str(case["case_id"])
        route = str(case["route"])
        week_of = pd.Timestamp(case["week_of"])

        expected_valid = bool(case["expected_valid"])
        expected_verdict = str(
            case["expected_verdict"]
        )

        note_id = case["note_id"]

        # ----------------------------------------------------
        # No-note test
        # ----------------------------------------------------

        if pd.isna(note_id) or str(note_id).strip() == "":

            actual_valid = False
            actual_verdict = "unexplained"

            note_found = False

        # ----------------------------------------------------
        # Specific-note test
        # ----------------------------------------------------

        else:

            note_id = str(note_id)

            matching_notes = notes[
                notes["note_id"].astype(str) == note_id
            ]

            if matching_notes.empty:

                actual_valid = False
                actual_verdict = "unexplained"
                note_found = False

            else:

                note = matching_notes.iloc[0]

                validation = validate_note(
                    note=note,
                    route=route,
                    week_of=week_of,
                )

                actual_valid = bool(
                    validation["valid"]
                )

                actual_verdict = str(
                    validation["verdict"]
                )

                note_found = True

        # ----------------------------------------------------
        # Compare expected vs actual
        # ----------------------------------------------------

        valid_match = (
            actual_valid == expected_valid
        )

        verdict_match = (
            actual_verdict == expected_verdict
        )

        passed = (
            valid_match
            and verdict_match
        )

        results.append(
            {
                "case_id": case_id,
                "route": route,
                "week_of": week_of.strftime(
                    "%Y-%m-%d"
                ),
                "note_id": (
                    ""
                    if pd.isna(case["note_id"])
                    else str(case["note_id"])
                ),
                "expected_valid": expected_valid,
                "actual_valid": actual_valid,
                "expected_verdict": expected_verdict,
                "actual_verdict": actual_verdict,
                "note_found": note_found,
                "passed": passed,
                "description": str(
                    case["description"]
                ),
            }
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    total_cases = len(results_df)

    passed_cases = int(
        results_df["passed"].sum()
    )

    failed_cases = (
        total_cases - passed_cases
    )

    accuracy = (
        passed_cases / total_cases
        if total_cases > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Build JSON result
    # --------------------------------------------------------

    evaluation_result = {
        "evaluation_name": (
            "FreightTiger Evidence Validation Evaluation"
        ),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": round(
            accuracy,
            4,
        ),
        "results": results,
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation_result,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n--- CASE RESULTS ---")

    for result in results:

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"{result['case_id']}: "
            f"{status} | "
            f"expected={result['expected_verdict']} | "
            f"actual={result['actual_verdict']}"
        )

    print("\n--- SUMMARY ---")

    print(
        f"Total cases: {total_cases}"
    )

    print(
        f"Passed: {passed_cases}"
    )

    print(
        f"Failed: {failed_cases}"
    )

    print(
        f"Pass rate: {accuracy:.2%}"
    )

    print(
        f"Results file: {RESULTS_FILE}"
    )

    print("=" * 70)

    return evaluation_result


if __name__ == "__main__":
    run_evaluation()