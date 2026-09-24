import pytest

from src.evidence_validator import (
    contains_positive_evidence,
    contains_negative_evidence,
    validate_route_match,
    validate_date_match,
    validate_note,
)


class TestPositiveEvidence:

    def test_surcharge_detected(self):
        assert contains_positive_evidence(
            "A temporary surcharge was applied due to high demand."
        )

    def test_higher_trip_costs_detected(self):
        assert contains_positive_evidence(
            "Flooding forced longer detours and higher trip costs."
        )

    def test_pushing_up_costs_detected(self):
        assert contains_positive_evidence(
            "Diesel prices rose, pushing up transportation costs."
        )

    def test_event_only_not_sufficient(self):
        # "Flooding" alone does not prove cost increase
        assert not contains_positive_evidence(
            "Heavy flooding disrupted highway traffic."
        )

    def test_normal_conditions_not_positive(self):
        assert not contains_positive_evidence(
            "Route returned to normal conditions after repairs."
        )


class TestNegativeEvidence:

    def test_costs_not_affected(self):
        assert contains_negative_evidence(
            "Costs were not significantly affected."
        )

    def test_no_major_disruptions(self):
        assert contains_negative_evidence(
            "No major disruptions reported this quarter."
        )

    def test_stable_demand(self):
        assert contains_negative_evidence(
            "Freight demand remained stable demand with normal operations."
        )

    def test_without_rate_change(self):
        assert contains_negative_evidence(
            "Compliance costs were absorbed without a rate change."
        )

    def test_cost_increase_not_negative(self):
        assert not contains_negative_evidence(
            "Higher trip costs were reported on this route."
        )


class TestRouteMatch:

    def test_exact_match(self):
        note = {"applies_to": "Mumbai-Pune"}
        assert validate_route_match(note, "Mumbai-Pune")

    def test_all_routes_matches(self):
        note = {"applies_to": "All Routes"}
        assert validate_route_match(note, "Delhi-Jaipur")

    def test_wrong_route_rejected(self):
        note = {"applies_to": "Ahmedabad-Mumbai"}
        assert not validate_route_match(note, "Mumbai-Pune")


class TestDateMatch:

    def test_same_date(self):
        note = {"date": "2025-01-20"}
        assert validate_date_match(note, "2025-01-20")

    def test_within_7_days(self):
        note = {"date": "2025-01-22"}
        assert validate_date_match(note, "2025-01-20")

    def test_beyond_7_days_rejected(self):
        note = {"date": "2025-02-01"}
        assert not validate_date_match(note, "2025-01-20")


class TestValidateNote:

    def test_valid_justified_note(self):
        note = {
            "note_id": "N002",
            "date": "2025-01-20",
            "applies_to": "Ahmedabad-Mumbai",
            "note": "A temporary surcharge applied due to high demand.",
        }
        result = validate_note(note, "Ahmedabad-Mumbai", "2025-01-20")
        assert result["valid"] == True
        assert result["verdict"] == "justified"

    def test_negative_evidence_rejects(self):
        note = {
            "note_id": "N005",
            "date": "2024-07-29",
            "applies_to": "Mumbai-Delhi",
            "note": "Minor delays; costs were not significantly affected.",
        }
        result = validate_note(note, "Mumbai-Delhi", "2024-07-29")
        assert result["valid"] == False
        assert result["verdict"] == "unexplained"

    def test_wrong_route_rejects(self):
        note = {
            "note_id": "N002",
            "date": "2025-01-20",
            "applies_to": "Ahmedabad-Mumbai",
            "note": "Temporary surcharge applied.",
        }
        result = validate_note(note, "Mumbai-Pune", "2025-01-20")
        assert result["valid"] == False

    def test_wrong_date_rejects(self):
        note = {
            "note_id": "N002",
            "date": "2025-06-01",
            "applies_to": "Ahmedabad-Mumbai",
            "note": "Temporary surcharge applied.",
        }
        result = validate_note(note, "Ahmedabad-Mumbai", "2025-01-20")
        assert result["valid"] == False
