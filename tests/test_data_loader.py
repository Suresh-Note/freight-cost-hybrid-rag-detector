import pandas as pd
import pytest

from src.data_loader import (
    load_shipments,
    load_context_notes,
    validate_columns,
)


class TestLoadShipments:

    def test_loads_dataframe(self):
        df = load_shipments()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_required_columns(self):
        df = load_shipments()
        required = [
            "shipment_id", "origin", "destination",
            "route_type", "quantity_tonnes", "distance_km",
            "freight_cost_inr", "shipment_date",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_shipment_date_is_datetime(self):
        df = load_shipments()
        assert pd.api.types.is_datetime64_any_dtype(df["shipment_date"])

    def test_numeric_columns_are_positive(self):
        df = load_shipments()
        assert (df["quantity_tonnes"] > 0).all()
        assert (df["distance_km"] > 0).all()
        assert (df["freight_cost_inr"] >= 0).all()


class TestLoadContextNotes:

    def test_loads_dataframe(self):
        df = load_context_notes()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_required_columns(self):
        df = load_context_notes()
        required = ["note_id", "date", "applies_to", "note"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_date_is_datetime(self):
        df = load_context_notes()
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


class TestValidateColumns:

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_columns(df, ["a", "b", "c"], "test.csv")

    def test_passes_when_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        validate_columns(df, ["a", "b"], "test.csv")
