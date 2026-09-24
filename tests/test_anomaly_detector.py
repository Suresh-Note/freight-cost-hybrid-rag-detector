import pandas as pd
import pytest

from src.anomaly_detector import detect_anomalies


def make_baseline_data():
    """Create data with known baselines for anomaly testing."""
    return pd.DataFrame({
        "route": ["A-B", "A-B", "C-D"],
        "week_of": pd.to_datetime([
            "2024-03-04", "2024-03-11", "2024-03-04",
        ]),
        "route_type": ["Short", "Short", "Short"],
        "cost_per_tonne_km": [3.0, 4.0, 2.5],
        "own_history_baseline": [2.5, 2.8, 2.4],
        "vs_own_history_pct": [20.0, 42.9, 4.2],
        "similar_route_baseline": [2.5, 2.5, 3.0],
        "vs_similar_routes_pct": [20.0, 60.0, -16.7],
    })


class TestAnomalyDetector:

    def test_flags_above_threshold(self):
        df = make_baseline_data()
        result = detect_anomalies(df)
        ab_week2 = result[
            (result["route"] == "A-B")
            & (result["week_of"] == pd.Timestamp("2024-03-11"))
        ].iloc[0]
        assert ab_week2["is_candidate_anomaly"] == True

    def test_does_not_flag_below_threshold(self):
        df = make_baseline_data()
        result = detect_anomalies(df)
        cd = result[result["route"] == "C-D"].iloc[0]
        assert cd["is_candidate_anomaly"] == False

    def test_flags_at_exact_threshold(self):
        df = make_baseline_data()
        result = detect_anomalies(df)
        # A-B week 1 has exactly 20% vs own history
        ab_week1 = result[
            (result["route"] == "A-B")
            & (result["week_of"] == pd.Timestamp("2024-03-04"))
        ].iloc[0]
        assert ab_week1["is_candidate_anomaly"] == True

    def test_or_logic(self):
        df = make_baseline_data()
        result = detect_anomalies(df)
        # is_candidate_anomaly should be OR of the two signals
        for _, row in result.iterrows():
            expected = (
                row["anomaly_own_history"]
                or row["anomaly_similar_routes"]
            )
            assert row["is_candidate_anomaly"] == expected
