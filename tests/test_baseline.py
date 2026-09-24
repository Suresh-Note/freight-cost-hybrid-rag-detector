import pandas as pd
import numpy as np
import pytest

from src.baseline import (
    calculate_own_history_baseline,
    calculate_similar_route_baseline,
)


def make_weekly_data():
    """Create weekly data spanning multiple weeks for baseline testing."""
    weeks = pd.date_range("2024-01-01", periods=10, freq="7D")

    rows = []
    for i, week in enumerate(weeks):
        # Route A: cost increases steadily
        rows.append({
            "route": "A-B",
            "week_of": week,
            "route_type": "Short",
            "cost_per_tonne_km": 2.0 + i * 0.1,
        })
        # Route C: stays flat
        rows.append({
            "route": "C-D",
            "week_of": week,
            "route_type": "Short",
            "cost_per_tonne_km": 2.5,
        })

    return pd.DataFrame(rows)


class TestOwnHistoryBaseline:

    def test_first_week_has_no_baseline(self):
        df = make_weekly_data()
        result = calculate_own_history_baseline(df)
        first_ab = result[result["route"] == "A-B"].iloc[0]
        assert pd.isna(first_ab["own_history_baseline"])

    def test_second_week_uses_first_week_only(self):
        df = make_weekly_data()
        result = calculate_own_history_baseline(df)
        ab_rows = result[result["route"] == "A-B"].sort_values("week_of")
        second = ab_rows.iloc[1]
        # Own history for week 2 should be week 1 value only
        assert abs(second["own_history_baseline"] - 2.0) < 0.01

    def test_no_lookahead(self):
        df = make_weekly_data()
        result = calculate_own_history_baseline(df)
        ab_rows = result[result["route"] == "A-B"].sort_values("week_of")

        for i in range(1, len(ab_rows)):
            row = ab_rows.iloc[i]
            # Baseline should never exceed the previous week's cost
            prev_cost = ab_rows.iloc[i - 1]["cost_per_tonne_km"]
            assert row["own_history_baseline"] <= prev_cost + 0.01

    def test_percentage_calculation(self):
        df = make_weekly_data()
        result = calculate_own_history_baseline(df)
        valid = result[result["vs_own_history_pct"].notna()]
        # Check percentage formula: (current - baseline) / baseline * 100
        for _, row in valid.iterrows():
            expected_pct = (
                (row["cost_per_tonne_km"] - row["own_history_baseline"])
                / row["own_history_baseline"]
            ) * 100
            assert abs(row["vs_own_history_pct"] - expected_pct) < 0.01


class TestSimilarRouteBaseline:

    def test_excludes_self(self):
        df = make_weekly_data()
        df = calculate_own_history_baseline(df)
        result = calculate_similar_route_baseline(df)

        # For route A-B, similar baseline should be C-D's cost (2.5)
        ab_rows = result[result["route"] == "A-B"]
        for _, row in ab_rows.iterrows():
            if pd.notna(row["similar_route_baseline"]):
                assert abs(row["similar_route_baseline"] - 2.5) < 0.01

    def test_single_route_type_has_peer(self):
        df = make_weekly_data()
        df = calculate_own_history_baseline(df)
        result = calculate_similar_route_baseline(df)
        # Both routes are "Short", so peers exist
        assert result["similar_route_baseline"].notna().any()
