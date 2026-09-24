from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.document_builder import (
    RAGDocument,
    load_rag_documents,
)


# Lightweight local embedding model.
# It runs on CPU and is downloaded/cached automatically
# on the first execution.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class VectorSearchResult:
    """
    Result returned by the dense vector store.
    """

    document: RAGDocument
    score: float


class LocalVectorStore:
    """
    Local semantic vector store.

    Architecture:

        RAG documents
             ↓
        Sentence Transformer
             ↓
        normalized embeddings
             ↓
        cosine similarity
             ↓
        top-k documents

    The embeddings are kept in memory for this project.
    This avoids requiring a native database dependency while
    preserving the dense-retrieval part of the RAG architecture.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.model_name = model_name

        print(
            f"Loading embedding model: {self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        self.documents = load_rag_documents()

        if not self.documents:
            raise ValueError(
                "No RAG documents were loaded."
            )

        self.document_texts = [
            document.text
            for document in self.documents
        ]

        print(
            f"Creating embeddings for "
            f"{len(self.documents)} documents..."
        )

        self.embeddings = self.model.encode(
            self.document_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self.embeddings = np.asarray(
            self.embeddings,
            dtype=np.float32,
        )

        if self.embeddings.ndim != 2:
            raise ValueError(
                "Document embeddings must be a 2D matrix."
            )

        print(
            f"Vector store ready: "
            f"{self.embeddings.shape[0]} documents x "
            f"{self.embeddings.shape[1]} dimensions"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """
        Perform semantic vector search.

        Because both document and query embeddings are
        normalized, dot product is equivalent to cosine
        similarity.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            results.append(
                VectorSearchResult(
                    document=self.documents[index],
                    score=float(scores[index]),
                )
            )

        return results


if __name__ == "__main__":

    print("=" * 70)
    print("LOCAL VECTOR STORE TEST")
    print("=" * 70)

    store = LocalVectorStore()

    test_query = (
        "Ahmedabad Mumbai festival surcharge "
        "high demand limited truck availability"
    )

    print(f"\nQuery:\n{test_query}")

    results = store.search(
        test_query,
        top_k=5,
    )

    print("\nTop semantic results:")
    print("-" * 70)

    for rank, result in enumerate(results, start=1):

        document = result.document

        print(
            f"{rank}. "
            f"{document.note_id} | "
            f"score={result.score:.4f} | "
            f"route={document.applies_to} | "
            f"date={document.date}"
        )

    print("=" * 70)