import pandas as pd

from src.config import SHIPMENT_FILE, CONTEXT_NOTES_FILE


# ============================================================
# REQUIRED COLUMNS
# ============================================================

SHIPMENT_REQUIRED_COLUMNS = [
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "material",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
    "transporter",
]

CONTEXT_REQUIRED_COLUMNS = [
    "note_id",
    "date",
    "applies_to",
    "note",
]


# ============================================================
# SHIPMENT DATA
# ============================================================

def load_shipments() -> pd.DataFrame:
    """
    Load and validate shipment_records.csv.
    """

    df = pd.read_csv(SHIPMENT_FILE)

    validate_columns(
        df,
        SHIPMENT_REQUIRED_COLUMNS,
        "shipment_records.csv",
    )

    # Convert numeric columns
    numeric_columns = [
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Convert date
    df["shipment_date"] = pd.to_datetime(
        df["shipment_date"],
        errors="coerce",
    )

    validate_shipments(df)

    return df


# ============================================================
# CONTEXT NOTES
# ============================================================

def load_context_notes() -> pd.DataFrame:
    """
    Load and validate context_notes.csv.
    """

    df = pd.read_csv(CONTEXT_NOTES_FILE)

    validate_columns(
        df,
        CONTEXT_REQUIRED_COLUMNS,
        "context_notes.csv",
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    validate_context_notes(df)

    return df


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    filename: str,
) -> None:
    """
    Make sure all required columns exist.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{filename} is missing required columns: "
            f"{missing_columns}"
        )


# ============================================================
# SHIPMENT VALIDATION
# ============================================================

def validate_shipments(df: pd.DataFrame) -> None:
    """
    Validate shipment data values.
    """

    if df.empty:
        raise ValueError("Shipment dataset is empty.")

    # Required fields should not be missing
    required_non_null = [
        "shipment_id",
        "origin",
        "destination",
        "route_type",
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
        "shipment_date",
    ]

    missing_counts = df[required_non_null].isna().sum()

    invalid_missing = missing_counts[
        missing_counts > 0
    ]

    if not invalid_missing.empty:
        raise ValueError(
            "Missing values found in shipment data:\n"
            f"{invalid_missing.to_dict()}"
        )

    # Quantity must be positive
    if (df["quantity_tonnes"] <= 0).any():
        raise ValueError(
            "quantity_tonnes must be greater than zero."
        )

    # Distance must be positive
    if (df["distance_km"] <= 0).any():
        raise ValueError(
            "distance_km must be greater than zero."
        )

    # Freight cost should not be negative
    if (df["freight_cost_inr"] < 0).any():
        raise ValueError(
            "freight_cost_inr cannot be negative."
        )


# ============================================================
# CONTEXT NOTE VALIDATION
# ============================================================

def validate_context_notes(df: pd.DataFrame) -> None:
    """
    Validate context note data.
    """

    if df.empty:
        raise ValueError("Context notes dataset is empty.")

    required_non_null = [
        "note_id",
        "date",
        "applies_to",
        "note",
    ]

    missing_counts = df[required_non_null].isna().sum()

    invalid_missing = missing_counts[
        missing_counts > 0
    ]

    if not invalid_missing.empty:
        raise ValueError(
            "Missing values found in context notes:\n"
            f"{invalid_missing.to_dict()}"
        )