from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "output" / "final_output.csv"
REPORT_FILE = PROJECT_ROOT / "logs" / "reproducibility_report.txt"

DETERMINISTIC_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
]

RUNS = 3


def run_pipeline(run_number):
    print(f"\n{'=' * 70}")
    print(f"REPRODUCIBILITY RUN {run_number}/{RUNS}")
    print(f"{'=' * 70}")

    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"Pipeline failed on reproducibility run {run_number}"
        )

    df = pd.read_csv(OUTPUT_FILE)

    return df


def compare_runs(run_a, run_b):
    if run_a.shape != run_b.shape:
        return False, f"Shape differs: {run_a.shape} vs {run_b.shape}"

    for column in DETERMINISTIC_COLUMNS:
        if column not in run_a.columns or column not in run_b.columns:
            return False, f"Missing deterministic column: {column}"

        if not run_a[column].equals(run_b[column]):
            differences = (run_a[column] != run_b[column]).sum()
            return False, (
                f"Column '{column}' differs in {differences} row(s)"
            )

    return True, "All deterministic fields match."


def main():
    print("=" * 70)
    print("FreightTiger Reproducibility Test")
    print("=" * 70)

    runs = []

    for run_number in range(1, RUNS + 1):
        df = run_pipeline(run_number)
        runs.append(df.copy())

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    comparison_results = []

    for i in range(len(runs) - 1):
        passed, message = compare_runs(runs[i], runs[i + 1])

        comparison_results.append(passed)

        status = "PASS" if passed else "FAIL"

        print(
            f"Run {i + 1} vs Run {i + 2}: "
            f"{status} | {message}"
        )

    all_passed = all(comparison_results)

    report_lines = [
        "FreightTiger Reproducibility Report",
        "=" * 50,
        "",
        f"Number of runs: {RUNS}",
        "Deterministic fields compared:",
    ]

    for column in DETERMINISTIC_COLUMNS:
        report_lines.append(f"- {column}")

    report_lines.extend([
        "",
        "Results:",
    ])

    for i, passed in enumerate(comparison_results, start=1):
        report_lines.append(
            f"- Run {i} vs Run {i + 1}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    report_lines.extend([
        "",
        f"Overall reproducibility: "
        f"{'PASS' if all_passed else 'FAIL'}",
        "",
        "Note:",
        "Explanation wording is intentionally excluded from the "
        "deterministic comparison because the case study allows "
        "explanation wording to vary.",
    ])

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)

    if all_passed:
        print("REPRODUCIBILITY TEST: PASS")
        print("All deterministic fields are identical across 3 runs.")
    else:
        print("REPRODUCIBILITY TEST: FAIL")
        print("Deterministic output changed between runs.")

    print(f"Report: {REPORT_FILE}")
    print("=" * 70)

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()