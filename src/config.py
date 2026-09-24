from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_DIR = PROJECT_ROOT / "eval"


SHIPMENT_FILE = DATA_DIR / "shipment_records.csv"
CONTEXT_NOTES_FILE = DATA_DIR / "context_notes.csv"
SAMPLE_OUTPUT_FILE = DATA_DIR / "sample_output_format_v2.csv"

FINAL_OUTPUT_FILE = OUTPUT_DIR / "final_output.csv"

TOKEN_COST_LOG_FILE = LOGS_DIR / "token_cost_log.json"
REPRODUCIBILITY_LOG_FILE = LOGS_DIR / "reproducibility_report.txt"


# ============================================================
# DATA PROCESSING
# ============================================================

# The case study requires Monday-Sunday weeks.
WEEK_START_DAY = "MONDAY"

# Own-history baseline:
# Use up to the previous 8 weeks only.
ROLLING_WINDOW_WEEKS = 8


# ============================================================
# CONTEXT NOTE RETRIEVAL
# ============================================================

# Candidate notes must be reasonably close to the shipment week.
NOTE_DATE_WINDOW_DAYS = 7


# ============================================================
# LLM CONFIGURATION
# ============================================================

# Primary provider
LLM_PROVIDER = "groq"

# Groq model used for grounded explanation generation.
GROQ_MODEL = "openai/gpt-oss-120b"

# Deterministic generation.
LLM_TEMPERATURE = 0.0

# Maximum completion budget.
# GPT-OSS reasoning can consume part of the completion budget,
# so 500 gives enough room for the final explanation.
LLM_MAX_OUTPUT_TOKENS = 500


# ============================================================
# REPRODUCIBILITY
# ============================================================

# Numerical decisions and flags must be deterministic.
RANDOM_SEED = 42


# ============================================================
# OUTPUT FORMAT
# ============================================================

OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


# ============================================================
# ANOMALY DETECTION
# ============================================================

# The case study does not prescribe a numeric threshold.
# We use 20% as an explicit, configurable engineering rule
# for identifying unusually high cost increases.
FLAG_THRESHOLD_PCT = 20.0