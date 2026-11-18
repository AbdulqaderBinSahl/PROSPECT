"""
Configuration parameters for the Step 1 corpus discovery pipeline.

The LLM endpoint (local server or cloud API) is resolved at runtime by
``common.env.resolve_llm_endpoint``. Scopus keys are read from the
environment, so ``common.env.load_env()`` must run before this module is
imported (``run_step1.py`` does this).
"""

from common.env import scopus_keys

# ---------------------------------------------------------------------------
# LLM generation settings (elicitation, strategy, query validation)
# ---------------------------------------------------------------------------
LLM_TEMPERATURE: float = 0.3
LLM_MAX_TOKENS: int = 4096

# ---------------------------------------------------------------------------
# Scopus API Configuration
# ---------------------------------------------------------------------------
SCOPUS_API_KEY, SCOPUS_INST_TOKEN = scopus_keys()  # institution token is optional
SCOPUS_BASE_URL: str = "https://api.elsevier.com/content/search/scopus"
SCOPUS_RESULTS_PER_PAGE: int = 25
SCOPUS_REQUEST_DELAY: float = 0.5  # seconds between API requests (rate limit)
SCOPUS_MAX_RESULTS_PER_QUERY: int = 2000  # safety cap per individual query

# ---------------------------------------------------------------------------
# Pipeline Parameters
# ---------------------------------------------------------------------------
MAX_CLARIFICATION_ROUNDS: int = 3
MAX_VALIDATION_ITERATIONS: int = 3
DEFAULT_TIME_WINDOW_YEARS: int = 10
LLM_JSON_RETRY_ATTEMPTS: int = 1  # retries if JSON parsing fails
