from pathlib import Path
import re

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

load_dotenv()

from src.config import (
    FINAL_OUTPUT_FILE,
    CONTEXT_NOTES_FILE,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)


class ShippingRAG:
    """
    Lightweight RAG assistant for the FreightTiger case study.

    Retrieval:
        TF-IDF + cosine similarity

    Knowledge sources:
        1. Final route-week findings
        2. Context notes

    Generation:
        Groq GPT-OSS 120B

    The LLM is instructed to answer only from retrieved evidence.
    """

    def __init__(self):
        self.output_df = pd.read_csv(FINAL_OUTPUT_FILE)
        self.notes_df = pd.read_csv(CONTEXT_NOTES_FILE)

        self.documents = []
        self.metadata = []

        self._build_documents()

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )

        self.document_matrix = self.vectorizer.fit_transform(
            self.documents
        )

        self.client = Groq()

    def _build_documents(self):
        """
        Convert findings and context notes into searchable documents.
        """

        # ---------------------------------------------------------
        # 1. Route-week findings
        # ---------------------------------------------------------
        for _, row in self.output_df.iterrows():

            route = str(row["route"])
            week = str(row["week_of"])
            cost = row["cost_per_tonne_km"]
            own = row["vs_own_history"]
            similar = row["vs_similar_routes"]
            flagged = str(row["flagged"])
            note_id = str(row["matched_note_id"])

            reason = str(row["reason"])

            document = (
                f"Route: {route}. "
                f"Week: {week}. "
                f"Cost per tonne-km: {cost}. "
                f"Comparison with own history: {own} percent. "
                f"Comparison with similar routes: {similar} percent. "
                f"Verdict: {flagged}. "
                f"Matched note: {note_id}. "
                f"Reason: {reason}."
            )

            self.documents.append(document)

            self.metadata.append(
                {
                    "source_type": "finding",
                    "route": route,
                    "week_of": week,
                    "note_id": note_id,
                }
            )

        # ---------------------------------------------------------
        # 2. Context notes
        # ---------------------------------------------------------
        for _, row in self.notes_df.iterrows():

            note_id = str(row["note_id"])
            date = str(row["date"])
            applies_to = str(row["applies_to"])
            note = str(row["note"])

            document = (
                f"Context note {note_id}. "
                f"Date: {date}. "
                f"Applies to: {applies_to}. "
                f"Note: {note}"
            )

            self.documents.append(document)

            self.metadata.append(
                {
                    "source_type": "context_note",
                    "route": applies_to,
                    "week_of": date,
                    "note_id": note_id,
                }
            )

    def retrieve(self, question, top_k=6):
        """
        Retrieve the most relevant findings and notes.
        """

        query_vector = self.vectorizer.transform([question])

        similarities = cosine_similarity(
            query_vector,
            self.document_matrix,
        )[0]

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            score = float(similarities[index])

            if score <= 0:
                continue

            results.append(
                {
                    "score": score,
                    "text": self.documents[index],
                    "metadata": self.metadata[index],
                }
            )

        return results

    def _build_prompt(self, question, retrieved):
        """
        Build a grounded RAG prompt.
        """

        evidence_blocks = []

        for i, item in enumerate(retrieved, start=1):

            metadata = item["metadata"]

            evidence_blocks.append(
                f"""
EVIDENCE {i}
Source type: {metadata["source_type"]}
Route: {metadata["route"]}
Week/date: {metadata["week_of"]}
Note ID: {metadata["note_id"]}
Content:
{item["text"]}
"""
            )

        evidence = "\n".join(evidence_blocks)

        prompt = f"""
You are the FreightTiger Smart Shipping Cost Assistant.

Answer the user's question using ONLY the evidence provided below.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence}

STRICT RULES:

1. Do not invent causes, numbers, dates, routes, or note IDs.

2. If the evidence does not establish a reason, explicitly say:
   "The available evidence does not establish a specific cause."

3. When discussing a cost increase, provide the relevant:
   - route
   - week/date
   - cost per tonne-km when available
   - comparison percentage when available
   - context note ID when applicable

4. A context note can explain a cost increase only when its content
   actually supports that conclusion.

5. A note that only mentions weather, flooding, repairs, disruption,
   festivals, fuel, tolls, or another event is NOT automatically proof
   that costs increased.

6. Do not treat a note saying costs were not affected as supporting
   evidence for a cost increase.

7. Distinguish between:
   - justified anomaly
   - unexplained anomaly
   - normal route-week

8. If multiple weeks are relevant, summarize them clearly.

9. Keep the answer concise and plain English.

10. Cite evidence naturally using note IDs such as [N001] and
    route/week information.

Now answer the question.
"""

        return prompt

    def ask(self, question, top_k=6):

        retrieved = self.retrieve(
            question,
            top_k=top_k,
        )

        if not retrieved:
            return {
                "answer": (
                    "I could not find relevant evidence in the "
                    "shipping findings or context notes."
                ),
                "retrieved": [],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            }

        prompt = self._build_prompt(
            question,
            retrieved,
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=LLM_MAX_OUTPUT_TOKENS,
            reasoning_effort="low",
            include_reasoning=False,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response.choices[0].message.content.strip()

        usage = response.usage

        return {
            "answer": answer,
            "retrieved": retrieved,
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }


def extract_question():
    """
    Supports:

        python -m src.rag_assistant "Why did Chennai-Bangalore get more expensive?"

    """

    import sys

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            'python -m src.rag_assistant "your question"'
        )
        raise SystemExit(1)

    return " ".join(sys.argv[1:])


if __name__ == "__main__":

    question = extract_question()

    rag = ShippingRAG()

    result = rag.ask(question)

    print("\n" + "=" * 70)
    print("FREIGHT TIGER RAG ASSISTANT")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved evidence:")
    print("-" * 70)

    for item in result["retrieved"]:

        metadata = item["metadata"]

        print(
            f'{metadata["source_type"]} | '
            f'route={metadata["route"]} | '
            f'date={metadata["week_of"]} | '
            f'note={metadata["note_id"]} | '
            f'score={item["score"]:.4f}'
        )

    print("\nToken usage:")
    print(
        f'Input: {result["usage"]["input_tokens"]} | '
        f'Output: {result["usage"]["output_tokens"]} | '
        f'Total: {result["usage"]["total_tokens"]}'
    )

    print("=" * 70)
