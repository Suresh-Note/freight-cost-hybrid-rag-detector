from dataclasses import dataclass
import re

from rank_bm25 import BM25Okapi

from src.rag.document_builder import (
    RAGDocument,
    load_rag_documents,
)


@dataclass(frozen=True)
class BM25SearchResult:
    """
    Result returned by BM25 lexical retrieval.
    """

    document: RAGDocument
    score: float


class BM25Retriever:
    """
    BM25 lexical retriever.

    BM25 complements dense semantic retrieval by matching
    important exact terms such as:
        - route names
        - note IDs
        - dates
        - event names
        - cost-impact phrases
    """

    def __init__(self):

        self.documents = load_rag_documents()

        if not self.documents:
            raise ValueError(
                "No RAG documents were loaded."
            )

        self.tokenized_documents = [
            self._tokenize(document.text)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple normalized tokenization.
        """

        text = text.lower()

        tokens = re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)*",
            text,
        )

        return tokens

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[BM25SearchResult]:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:

            results.append(
                BM25SearchResult(
                    document=self.documents[index],
                    score=float(scores[index]),
                )
            )

        return results


if __name__ == "__main__":

    retriever = BM25Retriever()

    query = (
        "Ahmedabad Mumbai festival surcharge "
        "high demand limited truck availability"
    )

    results = retriever.search(
        query,
        top_k=5,
    )

    print("=" * 70)
    print("BM25 RETRIEVAL TEST")
    print("=" * 70)

    print(f"Query:\n{query}\n")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        document = result.document

        print(
            f"{rank}. "
            f"{document.note_id} | "
            f"score={result.score:.4f} | "
            f"route={document.applies_to} | "
            f"date={document.date}"
        )

        print(
            f"   {document.text}"
        )

    print("=" * 70)