from dataclasses import dataclass

import numpy as np

from src.rag.bm25_retriever import BM25Retriever
from src.rag.document_builder import RAGDocument
from src.rag.vector_store import LocalVectorStore


DENSE_WEIGHT = 0.60
BM25_WEIGHT = 0.40


@dataclass(frozen=True)
class HybridSearchResult:
    """
    Combined result from dense + BM25 retrieval.
    """

    document: RAGDocument
    dense_score: float
    bm25_score: float
    dense_normalized: float
    bm25_normalized: float
    hybrid_score: float


class HybridRetriever:
    """
    Hybrid retrieval combining:

        1. Dense semantic retrieval
        2. BM25 lexical retrieval

    Scores are min-max normalized before combining.

    Final score:

        hybrid =
            0.60 * dense_score
            +
            0.40 * bm25_score
    """

    def __init__(
        self,
        dense_weight: float = DENSE_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ):

        if abs(
            dense_weight + bm25_weight - 1.0
        ) > 1e-9:
            raise ValueError(
                "Dense and BM25 weights must sum to 1.0."
            )

        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.dense_retriever = LocalVectorStore()
        self.bm25_retriever = BM25Retriever()

    @staticmethod
    def _normalize_scores(
        scores: list[float],
    ) -> list[float]:
        """
        Min-max normalize scores to [0, 1].

        If all scores are identical, assign 1.0 to each.
        """

        if not scores:
            return []

        values = np.asarray(
            scores,
            dtype=np.float64,
        )

        minimum = values.min()
        maximum = values.max()

        if np.isclose(
            minimum,
            maximum,
        ):
            return [1.0] * len(values)

        normalized = (
            (values - minimum)
            / (maximum - minimum)
        )

        return normalized.tolist()

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[HybridSearchResult]:
        """
        Retrieve candidates from both systems and merge them.

        candidate_k controls how many documents are considered
        from each retrieval system before hybrid scoring.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        dense_results = (
            self.dense_retriever.search(
                query,
                top_k=candidate_k,
            )
        )

        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=candidate_k,
            )
        )

        dense_scores = {
            result.document.document_id: result.score
            for result in dense_results
        }

        bm25_scores = {
            result.document.document_id: result.score
            for result in bm25_results
        }

        all_document_ids = (
            set(dense_scores)
            | set(bm25_scores)
        )

        documents = {}

        for result in dense_results:
            documents[
                result.document.document_id
            ] = result.document

        for result in bm25_results:
            documents[
                result.document.document_id
            ] = result.document

        dense_values = [
            dense_scores.get(
                document_id,
                0.0,
            )
            for document_id in all_document_ids
        ]

        bm25_values = [
            bm25_scores.get(
                document_id,
                0.0,
            )
            for document_id in all_document_ids
        ]

        normalized_dense_values = (
            self._normalize_scores(
                dense_values
            )
        )

        normalized_bm25_values = (
            self._normalize_scores(
                bm25_values
            )
        )

        results = []

        for position, document_id in enumerate(
            all_document_ids
        ):

            dense_normalized = (
                normalized_dense_values[position]
            )

            bm25_normalized = (
                normalized_bm25_values[position]
            )

            hybrid_score = (
                self.dense_weight
                * dense_normalized
                +
                self.bm25_weight
                * bm25_normalized
            )

            results.append(
                HybridSearchResult(
                    document=documents[document_id],
                    dense_score=dense_scores.get(
                        document_id,
                        0.0,
                    ),
                    bm25_score=bm25_scores.get(
                        document_id,
                        0.0,
                    ),
                    dense_normalized=dense_normalized,
                    bm25_normalized=bm25_normalized,
                    hybrid_score=hybrid_score,
                )
            )

        results.sort(
            key=lambda result: result.hybrid_score,
            reverse=True,
        )

        return results[:top_k]


if __name__ == "__main__":

    retriever = HybridRetriever()

    query = (
        "Ahmedabad Mumbai festival surcharge "
        "high demand limited truck availability"
    )

    results = retriever.search(
        query,
        top_k=5,
    )

    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    print(f"Query:\n{query}\n")

    print(
        f"Weights: "
        f"Dense={DENSE_WEIGHT:.2f}, "
        f"BM25={BM25_WEIGHT:.2f}\n"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        document = result.document

        print(
            f"{rank}. "
            f"{document.note_id} | "
            f"hybrid={result.hybrid_score:.4f} | "
            f"dense={result.dense_normalized:.4f} | "
            f"bm25={result.bm25_normalized:.4f} | "
            f"route={document.applies_to}"
        )

    print("=" * 70)