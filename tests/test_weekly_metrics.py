import pandas as pd
import pytest

from src.preprocessing import preprocess_shipments
from src.weekly_metrics import calculate_weekly_metrics


def make_sample_shipments():
    """Create minimal shipment data for testing."""
    return pd.DataFrame({
        "shipment_id": ["S1", "S2", "S3", "S4"],
        "origin": ["Mumbai", "Mumbai", "Delhi", "Delhi"],
        "destination": ["Pune", "Pune", "Jaipur", "Jaipur"],
        "route_type": ["Short", "Short", "Short", "Short"],
        "material": ["Steel", "Steel", "Cement", "Cement"],
        "quantity_tonnes": [10.0, 15.0, 20.0, 12.0],
        "distance_km": [150.0, 150.0, 280.0, 280.0],
        "freight_cost_inr": [5000, 7500, 16000, 10000],
        "shipment_date": pd.to_datetime([
            "2024-01-01", "2024-01-03",  # Same week
            "2024-01-01", "2024-01-05",  # Same week
        ]),
        "transporter": ["T1", "T1", "T2", "T2"],
    })


class TestWeeklyMetrics:

    def test_aggregates_by_route_and_week(self):
        df = preprocess_shipments(make_sample_shipments())
        weekly = calculate_weekly_metrics(df)

        # Two routes, one week each → 2 rows
        assert len(weekly) == 2

    def test_cost_per_tonne_km_formula(self):
        df = preprocess_shipments(make_sample_shipments())
        weekly = calculate_weekly_metrics(df)

        mumbai_pune = weekly[weekly["route"] == "Mumbai-Pune"].iloc[0]

        # total_cost = 5000 + 7500 = 12500
        # total_tonne_km = (10*150) + (15*150) = 1500 + 2250 = 3750
        # cost_per_tonne_km = 12500 / 3750 = 3.3333...
        expected = 12500 / 3750
        assert abs(mumbai_pune["cost_per_tonne_km"] - expected) < 0.01

    def test_has_required_output_columns(self):
        df = preprocess_shipments(make_sample_shipments())
        weekly = calculate_weekly_metrics(df)

        required = [
            "route", "week_of", "route_type",
            "cost_per_tonne_km", "total_tonne_km",
        ]
        for col in required:
            assert col in weekly.columns

    def test_no_missing_cost(self):
        df = preprocess_shipments(make_sample_shipments())
        weekly = calculate_weekly_metrics(df)
        assert not weekly["cost_per_tonne_km"].isna().any()
