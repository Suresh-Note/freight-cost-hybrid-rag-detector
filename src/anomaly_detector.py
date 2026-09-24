import pandas as pd

from src.config import FLAG_THRESHOLD_PCT


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify route-weeks with unusually high cost increases.

    A route-week is a candidate anomaly when either:

        vs_own_history_pct >= FLAG_THRESHOLD_PCT

    OR:

        vs_similar_routes_pct >= FLAG_THRESHOLD_PCT

    This function only identifies cost anomalies.

    It does NOT decide whether the anomaly is justified by
    a context note. That decision is handled later by the
    evidence validation stage.
    """

    required_columns = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "own_history_baseline",
        "vs_own_history_pct",
        "similar_route_baseline",
        "vs_similar_routes_pct",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weekly.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns for anomaly detection: "
            f"{missing_columns}"
        )

    data = weekly.copy()

    # --------------------------------------------------------
    # Positive increases only
    # --------------------------------------------------------

    own_history_signal = (
        data["vs_own_history_pct"]
        >= FLAG_THRESHOLD_PCT
    )

    similar_route_signal = (
        data["vs_similar_routes_pct"]
        >= FLAG_THRESHOLD_PCT
    )

    # --------------------------------------------------------
    # Candidate anomaly
    # --------------------------------------------------------

    data["anomaly_own_history"] = (
        own_history_signal
    )

    data["anomaly_similar_routes"] = (
        similar_route_signal
    )

    data["is_candidate_anomaly"] = (
        own_history_signal
        | similar_route_signal
    )

    return data


# ============================================================
# VALIDATION
# ============================================================

def validate_anomalies(
    weekly: pd.DataFrame,
) -> None:
    """
    Validate anomaly detection output.
    """

    required_columns = [
        "anomaly_own_history",
        "anomaly_similar_routes",
        "is_candidate_anomaly",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weekly.columns
    ]

    if missing_columns:
        raise ValueError(
            "Anomaly detection missing columns: "
            f"{missing_columns}"
        )

    # Candidate anomaly must equal the OR of the two signals.
    expected = (
        weekly["anomaly_own_history"]
        | weekly["anomaly_similar_routes"]
    )

    if not (
        weekly["is_candidate_anomaly"]
        == expected
    ).all():
        raise ValueError(
            "Candidate anomaly logic is inconsistent."
        )