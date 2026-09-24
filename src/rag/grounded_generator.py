from dataclasses import dataclass
import os

from dotenv import load_dotenv
from groq import Groq

from src.config import (
    GROQ_MODEL,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TEMPERATURE,
)


# Load environment variables from the project's .env file
load_dotenv()


@dataclass(frozen=True)
class GroundedExplanation:
    reason: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class GroundedGenerator:
    """
    Generates anomaly explanations using only validated RAG evidence.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. "
                "Add GROQ_API_KEY=your_api_key to the project's .env file."
            )

        self.client = Groq(api_key=api_key)

    @staticmethod
    def _build_prompt(
        route: str,
        week_of: str,
        cost_per_tonne_km: float,
        vs_own_history: float | None,
        vs_similar_routes: float | None,
        note_id: str,
        note_date: str,
        note_text: str,
    ) -> str:

        own_history_text = (
            f"{vs_own_history:.1f}%"
            if vs_own_history is not None
            else "N/A"
        )

        similar_routes_text = (
            f"{vs_similar_routes:.1f}%"
            if vs_similar_routes is not None
            else "N/A"
        )

        return f"""
You are generating a grounded explanation for a shipment-cost anomaly.

STRICT RULES:
1. Use ONLY the supplied numerical metrics and the supplied context note.
2. Do NOT invent causes, events, weather conditions, market conditions,
   operational problems, or business reasons.
3. Do NOT add facts that are not explicitly supported by the note.
4. The explanation MUST mention the exact note ID.
5. The explanation MUST mention the applicable route.
6. The explanation MUST mention the applicable week.
7. Explain why the supplied note supports the observed cost movement.
8. Keep the explanation concise: 1-2 sentences.
9. Do not claim that the note proves anything beyond what it explicitly states.
10. Do not mention information from your general knowledge.
11. Do not use another note as supporting evidence.
12. Do not change the supplied numerical values.

ANOMALY DATA:
Route: {route}
Week: {week_of}
Cost per tonne-km: {cost_per_tonne_km:.2f}
Vs own 8-week history: {own_history_text}
Vs similar routes: {similar_routes_text}

VALIDATED CONTEXT NOTE:
Note ID: {note_id}
Note date: {note_date}
Note applies to: {route}
Note content:
{note_text}

Write ONLY the final explanation.
""".strip()

    def generate(
        self,
        route: str,
        week_of: str,
        cost_per_tonne_km: float,
        vs_own_history: float | None,
        vs_similar_routes: float | None,
        note_id: str,
        note_date: str,
        note_text: str,
    ) -> GroundedExplanation:

        prompt = self._build_prompt(
            route=route,
            week_of=week_of,
            cost_per_tonne_km=cost_per_tonne_km,
            vs_own_history=vs_own_history,
            vs_similar_routes=vs_similar_routes,
            note_id=note_id,
            note_date=note_date,
            note_text=note_text,
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

        choice = response.choices[0]

        reason = (choice.message.content or "").strip()

        if not reason:
            raise ValueError(
                "Groq returned an empty explanation."
            )

        usage = response.usage

        input_tokens = getattr(
            usage,
            "prompt_tokens",
            0,
        ) or 0

        output_tokens = getattr(
            usage,
            "completion_tokens",
            0,
        ) or 0

        total_tokens = getattr(
            usage,
            "total_tokens",
            0,
        ) or 0

        return GroundedExplanation(
            reason=reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("GROUNDED GENERATOR TEST")
    print("=" * 70)

    generator = GroundedGenerator()

    result = generator.generate(
        route="Ahmedabad-Mumbai",
        week_of="2025-01-20",
        cost_per_tonne_km=3.20,
        vs_own_history=29.5,
        vs_similar_routes=22.5,
        note_id="N002",
        note_date="2025-01-20",
        note_text=(
            "A regional festival week saw a temporary surcharge applied "
            "by transporters on the Ahmedabad-Mumbai corridor due to high "
            "demand and limited truck availability."
        ),
    )

    print("\nGenerated explanation:")
    print("-" * 70)
    print(result.reason)

    print("\nToken usage:")
    print("-" * 70)
    print(f"Input tokens : {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Total tokens : {result.total_tokens}")

    print("=" * 70)