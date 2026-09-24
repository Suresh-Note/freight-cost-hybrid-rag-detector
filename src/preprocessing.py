import pandas as pd


# ============================================================
# ROUTE CREATION
# ============================================================

def create_route(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create route using origin and destination.

    Example:
        Delhi + Jaipur -> Delhi-Jaipur

    route_type is NOT calculated here because the case study
    provides route_type and requires us to use that value.
    """

    df = df.copy()

    df["origin"] = df["origin"].astype(str).str.strip()
    df["destination"] = df["destination"].astype(str).str.strip()

    df["route"] = (
        df["origin"] + "-" + df["destination"]
    )

    return df


# ============================================================
# WEEK CREATION
# ============================================================

def create_week_of(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert shipment_date into Monday-based weeks.

    The case study defines weeks as Monday-Sunday.

    week_of = Monday date of that week.
    """

    df = df.copy()

    df["shipment_date"] = pd.to_datetime(
        df["shipment_date"],
        errors="raise",
    )

    # Monday = 0, Tuesday = 1, ..., Sunday = 6
    df["week_of"] = (
        df["shipment_date"]
        - pd.to_timedelta(
            df["shipment_date"].dt.weekday,
            unit="D",
        )
    )

    # Keep only the date portion.
    df["week_of"] = df["week_of"].dt.normalize()

    return df


# ============================================================
# COMPLETE PREPROCESSING
# ============================================================

def preprocess_shipments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all shipment preprocessing steps.

    Steps:
        1. Create route.
        2. Create Monday-based week_of.
        3. Preserve provided route_type.
    """

    df = df.copy()

    df = create_route(df)

    df = create_week_of(df)

    return df


# ============================================================
# BASIC PREPROCESSING VALIDATION
# ============================================================

def validate_preprocessed_data(df: pd.DataFrame) -> None:
    """
    Validate the fields created during preprocessing.
    """

    required_columns = [
        "route",
        "route_type",
        "week_of",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Preprocessed data is missing columns: "
            f"{missing_columns}"
        )

    if df["route"].isna().any():
        raise ValueError(
            "Route contains missing values."
        )

    if df["week_of"].isna().any():
        raise ValueError(
            "week_of contains missing values."
        )

    # Every week_of must be a Monday.
    if not (df["week_of"].dt.weekday == 0).all():
        raise ValueError(
            "week_of contains dates that are not Mondays."
        )