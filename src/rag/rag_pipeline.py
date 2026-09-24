from dataclasses import dataclass
from datetime import datetime

from src.evidence_validator import validate_note
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import (
    CrossEncoderReranker,
)


DATE_WINDOW_DAYS = 7


@dataclass(frozen=True)
class RAGEvidence:
    note_id: str
    note_date: str
    applies_to: str
    note_text: str
    rerank_score: float
    route_match: bool
    date_match: bool
    evidence_valid: bool


class RAGPipeline:
    def __init__(
        self,
        date_window_days: int = DATE_WINDOW_DAYS,
    ):
        self.date_window_days = date_window_days

        self.hybrid_retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()

    @staticmethod
    def _parse_date(value: str):
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()

    def _route_matches(
        self,
        note_route: str,
        target_route: str,
    ) -> bool:

        note_route = note_route.strip()
        target_route = target_route.strip()

        return (
            note_route == target_route
            or note_route.lower() == "all routes"
        )

    def _date_matches(
        self,
        note_date: str,
        target_week: str,
    ) -> bool:

        note_dt = self._parse_date(note_date)
        target_dt = self._parse_date(target_week)

        difference = abs(
            (note_dt - target_dt).days
        )

        return difference <= self.date_window_days

    def retrieve_evidence(
        self,
        query: str,
        route: str,
        week_of: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[RAGEvidence]:

        candidates = self.hybrid_retriever.search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )

        evidence = []

        for result in reranked:

            document = result.document

            route_match = self._route_matches(
                document.applies_to,
                route,
            )

            date_match = self._date_matches(
                document.date,
                week_of,
            )

            note = {
                "note_id": document.note_id,
                "date": document.date,
                "applies_to": document.applies_to,
                "note": document.text,
            }

            evidence_valid = False

            if route_match and date_match:

                validation = validate_note(
                    note,
                    route,
                    week_of,
                )

                # IMPORTANT:
                # validate_note() returns a dictionary.
                # We must use its explicit "valid" field.
                if isinstance(validation, dict):
                    evidence_valid = bool(
                        validation.get(
                            "valid",
                            False,
                        )
                    )
                else:
                    evidence_valid = bool(
                        validation
                    )

            evidence.append(
                RAGEvidence(
                    note_id=document.note_id,
                    note_date=document.date,
                    applies_to=document.applies_to,
                    note_text=document.text,
                    rerank_score=result.rerank_score,
                    route_match=route_match,
                    date_match=date_match,
                    evidence_valid=evidence_valid,
                )
            )

        return evidence

    def find_valid_evidence(
        self,
        query: str,
        route: str,
        week_of: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> RAGEvidence | None:

        evidence = self.retrieve_evidence(
            query=query,
            route=route,
            week_of=week_of,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        for item in evidence:

            if item.evidence_valid:
                return item

        return None


if __name__ == "__main__":

    pipeline = RAGPipeline()

    query = (
        "Ahmedabad Mumbai cost increase "
        "festival surcharge high demand "
        "limited truck availability"
    )

    route = "Ahmedabad-Mumbai"
    week_of = "2025-01-20"

    print("=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)

    print(f"Route: {route}")
    print(f"Week: {week_of}")
    print(f"Query: {query}")

    results = pipeline.retrieve_evidence(
        query=query,
        route=route,
        week_of=week_of,
        top_k=5,
    )

    print("\nRetrieved and validated evidence:")
    print("-" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result.note_id} | "
            f"rerank={result.rerank_score:.4f} | "
            f"route_match={result.route_match} | "
            f"date_match={result.date_match} | "
            f"valid={result.evidence_valid}"
        )

    valid = pipeline.find_valid_evidence(
        query=query,
        route=route,
        week_of=week_of,
    )

    print("\nFinal evidence:")
    print("-" * 70)

    if valid:

        print(
            f"VALID: {valid.note_id} | "
            f"{valid.note_text}"
        )

    else:

        print(
            "No valid supporting evidence found."
        )

    print("=" * 70)