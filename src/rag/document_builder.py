from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import CONTEXT_NOTES_FILE


@dataclass(frozen=True)
class RAGDocument:
    """
    Structured document used by the RAG retrieval layer.
    """

    document_id: str
    note_id: str
    date: str
    applies_to: str
    text: str


def load_rag_documents(
    notes_file: Path = CONTEXT_NOTES_FILE,
) -> list[RAGDocument]:
    """
    Load context notes and convert them into RAG documents.

    Each context note remains traceable to its original:
        note_id
        date
        applies_to
        note text
    """

    df = pd.read_csv(notes_file)

    required_columns = {
        "note_id",
        "date",
        "applies_to",
        "note",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Context notes missing required columns: {sorted(missing)}"
        )

    documents = []

    for _, row in df.iterrows():

        note_id = str(row["note_id"]).strip()
        date = str(row["date"]).strip()
        applies_to = str(row["applies_to"]).strip()
        note = str(row["note"]).strip()

        searchable_text = (
            f"Note ID: {note_id}. "
            f"Date: {date}. "
            f"Applies to: {applies_to}. "
            f"Context: {note}"
        )

        documents.append(
            RAGDocument(
                document_id=f"context_{note_id}",
                note_id=note_id,
                date=date,
                applies_to=applies_to,
                text=searchable_text,
            )
        )

    return documents


if __name__ == "__main__":

    documents = load_rag_documents()

    print("=" * 70)
    print("RAG DOCUMENT BUILDER")
    print("=" * 70)

    print(f"Documents loaded: {len(documents)}")

    for document in documents:
        print(
            f"\n{document.document_id}"
            f" | date={document.date}"
            f" | applies_to={document.applies_to}"
        )
        print(f"  {document.text}")

    print("=" * 70)