import re

import pandas as pd


# ============================================================
# POSITIVE / NEGATIVE EVIDENCE TERMS
# ============================================================

# IMPORTANT:
# These patterns require explicit language connecting an event
# to a freight/shipping/transportation cost impact.
#
# Event-only words such as "flood", "weather", "toll", or
# "disruption" are intentionally NOT sufficient by themselves.

POSITIVE_REASON_PATTERNS = [
    # Direct cost increase language
    r"cost increase",
    r"cost increased",
    r"costs increased",
    r"cost rose",
    r"costs rose",
    r"higher cost",
    r"higher costs",
    r"higher freight",
    r"higher trip costs",
    r"increased transportation costs",
    r"pushing up transportation costs",
    r"pushed up transportation costs",
    r"transportation costs rose",
    r"transportation cost rose",
    r"freight costs rose",
    r"freight cost rose",
    r"freight costs increased",
    r"freight cost increased",

    # Surcharge / rate language
    r"temporary surcharge",
    r"surcharge applied",
    r"surcharge",
    r"price increase",
    r"rate increase",
    r"rate increased",
    r"rates increased",

    # Fuel-related explicit cost impact
    r"fuel prices? .* (increased|rose|pushed up|increasing)",
    r"fuel .* pushing up .* costs",
    r"fuel .* increased .* costs",

    # Toll-related explicit cost impact
    r"toll increase",
    r"toll .* increased .* costs",
    r"toll .* pushed up .* costs",

    # Demand / availability explicitly tied to surcharge/cost
    r"due to high demand",
    r"due to increased demand",
    r"due to limited truck availability",
    r"due to truck shortage",
    r"limited truck availability.*surcharge",
    r"limited truck availability.*cost",
]


NEGATIVE_REASON_PATTERNS = [
    # Explicit no-cost-impact statements
    r"costs were not affected",
    r"cost was not affected",
    r"costs were not significantly affected",
    r"cost was not significantly affected",
    r"no impact on cost",
    r"no impact on costs",
    r"did not affect costs",
    r"did not affect cost",
    r"without a rate change",
    r"without rate change",

    # No surcharge / increase
    r"no surcharge",
    r"no increase",

    # No disruption / stable conditions
    r"no major disruptions",
    r"no major disruption",
    r"no significant disruptions",
    r"no significant disruption",
    r"stable demand",

    # Explicitly outside the dataset
    r"affected routes are not part of this dataset",

    # Cost absorbed / no customer freight-rate impact
    r"compliance costs were absorbed",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic matching.
    """

    text = str(text).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ============================================================
# NEGATIVE EVIDENCE CHECK
# ============================================================

def contains_negative_evidence(
    note: str,
) -> bool:
    """
    Return True when the note explicitly indicates that
    the event did not establish a relevant cost impact,
    had no relevant disruption, or does not apply to
    the dataset.
    """

    text = normalize_text(note)

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern in NEGATIVE_REASON_PATTERNS
    )


# ============================================================
# POSITIVE EVIDENCE CHECK
# ============================================================

def contains_positive_evidence(
    note: str,
) -> bool:
    """
    Return True only when the note contains explicit language
    connecting an event to a freight/shipping/transportation
    cost increase or surcharge.

    Event-only words such as:
        flood
        weather
        toll
        disruption

    are intentionally insufficient.
    """

    text = normalize_text(note)

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern in POSITIVE_REASON_PATTERNS
    )


# ============================================================
# ROUTE VALIDATION
# ============================================================

def validate_route_match(
    note: pd.Series,
    route: str,
) -> bool:
    """
    Verify that the note applies to the route.

    'All Routes' is considered applicable unless the note itself
    contains explicit evidence that the affected routes are
    outside this dataset.
    """

    applies_to = normalize_text(
        note["applies_to"]
    )

    route = normalize_text(route)

    return (
        applies_to == "all routes"
        or applies_to == route
    )


# ============================================================
# DATE VALIDATION
# ============================================================

def validate_date_match(
    note: pd.Series,
    week_of,
    date_window_days: int = 7,
) -> bool:
    """
    Verify that the note date falls within the allowed
    time window around the route-week.

    The case-study implementation uses a ±7-day evidence
    window.
    """

    note_date = pd.Timestamp(
        note["date"]
    )

    week_of = pd.Timestamp(
        week_of
    )

    difference = abs(
        (note_date - week_of).days
    )

    return difference <= date_window_days


# ============================================================
# COMPLETE EVIDENCE VALIDATION
# ============================================================

def validate_note(
    note: pd.Series,
    route: str,
    week_of,
    date_window_days: int = 7,
) -> dict:
    """
    Determine whether a retrieved note provides valid evidence
    for a route-week cost increase.

    A note is valid only when:

        1. Route matches.
        2. Date matches.
        3. It contains explicit cost-impact evidence.
        4. It does NOT contain explicit negative evidence.

    Event-only words such as "flood", "weather", "toll", or
    "disruption" do not qualify by themselves.

    Returns a structured validation result.
    """

    route_valid = validate_route_match(
        note,
        route,
    )

    date_valid = validate_date_match(
        note,
        week_of,
        date_window_days,
    )

    negative_evidence = contains_negative_evidence(
        note["note"]
    )

    positive_evidence = contains_positive_evidence(
        note["note"]
    )

    valid = (
        route_valid
        and date_valid
        and positive_evidence
        and not negative_evidence
    )

    if valid:
        verdict = "justified"
    else:
        verdict = "unexplained"

    return {
        "valid": valid,
        "verdict": verdict,
        "route_valid": route_valid,
        "date_valid": date_valid,
        "positive_evidence": positive_evidence,
        "negative_evidence": negative_evidence,
        "note_id": note["note_id"],
    }


# ============================================================
# FIND FIRST VALID NOTE
# ============================================================

def find_valid_note(
    candidates: pd.DataFrame,
    route: str,
    week_of,
    date_window_days: int = 7,
):
    """
    Examine retrieved candidate notes in ranking order and
    return the first note that passes evidence validation.

    Returns:
        note, validation_result

    If no note passes:
        None, None
    """

    for _, note in candidates.iterrows():

        validation = validate_note(
            note=note,
            route=route,
            week_of=week_of,
            date_window_days=date_window_days,
        )

        if validation["valid"]:
            return note, validation

    return None, None