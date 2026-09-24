import pandas as pd


# ============================================================
# WEEKLY COST PER TONNE-KM
# ============================================================

def calculate_weekly_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate weekly shipping cost per tonne-km.

    Formula:

        total freight cost
        -------------------------
        total tonne-km

    where:

        tonne-km = quantity_tonnes × distance_km

    Aggregation is performed by:
        route + week_of

    route_type is preserved from the source data.
    """

    required_columns = [
        "route",
        "week_of",
        "route_type",
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for weekly metrics: "
            f"{missing_columns}"
        )

    data = df.copy()

    # --------------------------------------------------------
    # Calculate tonne-km for every shipment
    # --------------------------------------------------------

    data["tonne_km"] = (
        data["quantity_tonnes"]
        * data["distance_km"]
    )

    # --------------------------------------------------------
    # Aggregate by route and week
    # --------------------------------------------------------

    weekly = (
        data.groupby(
            ["route", "week_of"],
            as_index=False,
        )
        .agg(
            route_type=("route_type", "first"),
            total_quantity_tonnes=(
                "quantity_tonnes",
                "sum",
            ),
            total_distance_km=(
                "distance_km",
                "sum",
            ),
            total_tonne_km=(
                "tonne_km",
                "sum",
            ),
            total_freight_cost_inr=(
                "freight_cost_inr",
                "sum",
            ),
            shipment_count=(
                "route",
                "size",
            ),
        )
    )

    # --------------------------------------------------------
    # Cost per tonne-km
    # --------------------------------------------------------

    weekly["cost_per_tonne_km"] = (
        weekly["total_freight_cost_inr"]
        / weekly["total_tonne_km"]
    )

    # --------------------------------------------------------
    # Sort for deterministic downstream processing
    # --------------------------------------------------------

    weekly = weekly.sort_values(
        ["route", "week_of"]
    ).reset_index(drop=True)

    return weekly


# ============================================================
# VALIDATION
# ============================================================

def validate_weekly_metrics(
    weekly: pd.DataFrame,
) -> None:
    """
    Validate the calculated weekly metrics.
    """

    required_columns = [
        "route",
        "week_of",
        "route_type",
        "total_quantity_tonnes",
        "total_tonne_km",
        "total_freight_cost_inr",
        "cost_per_tonne_km",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weekly.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Weekly metrics missing columns: "
            f"{missing_columns}"
        )

    if weekly.empty:
        raise ValueError(
            "Weekly metrics dataframe is empty."
        )

    if weekly["cost_per_tonne_km"].isna().any():
        raise ValueError(
            "cost_per_tonne_km contains missing values."
        )

    if weekly["cost_per_tonne_km"].isin(
        [float("inf"), float("-inf")]
    ).any():
        raise ValueError(
            "cost_per_tonne_km contains infinite values."
        )

    if (weekly["total_tonne_km"] <= 0).any():
        raise ValueError(
            "total_tonne_km must be greater than zero."
        )