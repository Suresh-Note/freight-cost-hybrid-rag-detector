import pandas as pd

from src.config import ROLLING_WINDOW_WEEKS


# ============================================================
# OWN-HISTORY BASELINE
# ============================================================

def calculate_own_history_baseline(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the trailing own-history average for each
    route-week.

    Rules:
        - Use only weeks strictly BEFORE the current week.
        - Use up to the previous 8 weeks.
        - If fewer than 8 previous weeks exist, use all
          available previous weeks.
        - Never use the current week's value.
        - Never use future weeks.
    """

    required_columns = [
        "route",
        "week_of",
        "cost_per_tonne_km",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weekly.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns for own-history baseline: "
            f"{missing_columns}"
        )

    data = weekly.copy()

    data["week_of"] = pd.to_datetime(
        data["week_of"]
    )

    data = data.sort_values(
        ["route", "week_of"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Calculate baseline separately for each route
    # --------------------------------------------------------

    data["own_history_baseline"] = (
        data.groupby("route")["cost_per_tonne_km"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window=ROLLING_WINDOW_WEEKS,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    # --------------------------------------------------------
    # Percentage difference from own history
    # --------------------------------------------------------

    data["vs_own_history_pct"] = (
        (
            data["cost_per_tonne_km"]
            - data["own_history_baseline"]
        )
        / data["own_history_baseline"]
    ) * 100

    return data


# ============================================================
# OWN-HISTORY VALIDATION
# ============================================================

def validate_own_history_baseline(
    weekly: pd.DataFrame,
) -> None:
    """
    Validate own-history baseline calculations.
    """

    required_columns = [
        "own_history_baseline",
        "vs_own_history_pct",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weekly.columns
    ]

    if missing_columns:
        raise ValueError(
            "Own-history baseline missing columns: "
            f"{missing_columns}"
        )

    # First week of every route must have no baseline,
    # because there are no previous weeks.
    first_rows = (
        weekly.sort_values(
            ["route", "week_of"]
        )
        .groupby("route")
        .head(1)
    )

    if first_rows["own_history_baseline"].notna().any():
        raise ValueError(
            "First week of a route must not have "
            "an own-history baseline."
        )

    # Every available baseline must be positive.
    valid_baselines = weekly[
        weekly["own_history_baseline"].notna()
    ]["own_history_baseline"]

    if (valid_baselines <= 0).any():
        raise ValueError(
            "Own-history baseline must be greater than zero."
        )


# ============================================================
# SIMILAR-ROUTE BASELINE
# ============================================================

def calculate_similar_route_baseline(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the same-week average cost for similar routes.

    Similar routes are:
        - Same week_of
        - Same route_type
        - Different route

    The current route is excluded from the comparison.
    """

    required_columns = [
        "route",
        "week_of",
        "route_type",
        "cost_per_tonne_km",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weekly.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns for similar-route baseline: "
            f"{missing_columns}"
        )

    data = weekly.copy()

    data["week_of"] = pd.to_datetime(
        data["week_of"]
    )

    # --------------------------------------------------------
    # Group by same week + same route type
    # --------------------------------------------------------

    group_columns = [
        "week_of",
        "route_type",
    ]

    data["peer_cost_sum"] = (
        data.groupby(group_columns)["cost_per_tonne_km"]
        .transform("sum")
    )

    data["peer_route_count"] = (
        data.groupby(group_columns)["cost_per_tonne_km"]
        .transform("count")
    )

    # --------------------------------------------------------
    # Exclude the current route itself
    # --------------------------------------------------------

    other_route_count = (
        data["peer_route_count"] - 1
    )

    other_route_cost_sum = (
        data["peer_cost_sum"]
        - data["cost_per_tonne_km"]
    )

    data["similar_route_baseline"] = (
        other_route_cost_sum
        / other_route_count
    )

    # If there are no other routes of the same route_type
    # in the same week, no peer baseline exists.
    data.loc[
        other_route_count <= 0,
        "similar_route_baseline",
    ] = float("nan")

    # --------------------------------------------------------
    # Percentage difference from similar routes
    # --------------------------------------------------------

    data["vs_similar_routes_pct"] = (
        (
            data["cost_per_tonne_km"]
            - data["similar_route_baseline"]
        )
        / data["similar_route_baseline"]
    ) * 100

    # --------------------------------------------------------
    # Remove temporary columns
    # --------------------------------------------------------

    data = data.drop(
        columns=[
            "peer_cost_sum",
            "peer_route_count",
        ]
    )

    return data


# ============================================================
# SIMILAR-ROUTE VALIDATION
# ============================================================

def validate_similar_route_baseline(
    weekly: pd.DataFrame,
) -> None:
    """
    Validate similar-route baseline calculations.
    """

    required_columns = [
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
            "Similar-route baseline missing columns: "
            f"{missing_columns}"
        )

    # Any available peer baseline must be positive.
    valid_baselines = weekly[
        weekly["similar_route_baseline"].notna()
    ]["similar_route_baseline"]

    if (valid_baselines <= 0).any():
        raise ValueError(
            "Similar-route baseline must be greater than zero."
        )