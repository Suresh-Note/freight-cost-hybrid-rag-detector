from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from src.rag.document_builder import RAGDocument
from src.rag.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
)


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankedResult:
    """
    Result after cross-encoder reranking.
    """

    document: RAGDocument
    rerank_score: float

    dense_score: float
    bm25_score: float
    hybrid_score: float


class CrossEncoderReranker:
    """
    Cross-encoder reranker.

    The model receives:

        [query, candidate document]

    together and directly estimates their relevance.

    This is more precise than comparing independent
    embedding vectors.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
    ):
        self.model_name = model_name

        print(
            f"Loading reranker model: {self.model_name}"
        )

        self.model = CrossEncoder(
            self.model_name
        )

    def rerank(
        self,
        query: str,
        candidates: list[HybridSearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not candidates:
            return []

        pairs = [
            [
                query,
                candidate.document.text,
            ]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        results = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            results.append(
                RerankedResult(
                    document=candidate.document,
                    rerank_score=float(score),
                    dense_score=candidate.dense_score,
                    bm25_score=candidate.bm25_score,
                    hybrid_score=candidate.hybrid_score,
                )
            )

        results.sort(
            key=lambda result: result.rerank_score,
            reverse=True,
        )

        return results[:top_k]


if __name__ == "__main__":

    query = (
        "Ahmedabad Mumbai festival surcharge "
        "high demand limited truck availability"
    )

    print("=" * 70)
    print("CROSS-ENCODER RERANKER TEST")
    print("=" * 70)

    hybrid_retriever = HybridRetriever()

    candidates = hybrid_retriever.search(
        query,
        top_k=5,
        candidate_k=10,
    )

    reranker = CrossEncoderReranker()

    results = reranker.rerank(
        query,
        candidates,
        top_k=5,
    )

    print(
        f"\nQuery:\n{query}\n"
    )

    print("Reranked results:")
    print("-" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        document = result.document

        print(
            f"{rank}. "
            f"{document.note_id} | "
            f"rerank={result.rerank_score:.4f} | "
            f"hybrid={result.hybrid_score:.4f} | "
            f"route={document.applies_to} | "
            f"date={document.date}"
        )

    print("=" * 70)