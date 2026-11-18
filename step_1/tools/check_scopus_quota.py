"""
Check your Scopus API key quota and reset time with one minimal request.

Keys are read from .env (SCOPUS_API_KEY, optional SCOPUS_API_INSTITUTION_KEY).

Usage (from the repository root):
    python step_1/tools/check_scopus_quota.py
"""

import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.env import load_env, scopus_keys  # noqa: E402

# Shell variables take precedence over .env; remember which was used (never the value).
_SHELL_VARS = {name for name in os.environ if name.startswith("SCOPUS")}
load_env()
API_KEY, INST_TOKEN = scopus_keys()


def _where(name: str) -> str:
    return "shell environment (overrides .env)" if name in _SHELL_VARS else ".env"


def _token_source() -> str:
    for name in ("SCOPUS_API_INSTITUTION_KEY", "SCOPUS_INST_TOKEN"):
        if os.getenv(name):
            return f"yes, from {name} in the {_where(name)}"
    return "no"

SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"


def check_quota():
    if not API_KEY:
        print("ERROR: No SCOPUS_API_KEY found. Add it to .env (see .env.example).")
        return

    headers = {
        "X-ELS-APIKey": API_KEY,
        "Accept": "application/json",
    }
    if INST_TOKEN:
        headers["X-ELS-Insttoken"] = INST_TOKEN

    # Make a minimal search request to get rate-limit headers back
    params = {
        "query": "TITLE(test)",
        "count": 1,
        "start": 0,
    }

    print(f"SCOPUS_API_KEY read from: the {_where('SCOPUS_API_KEY')}")
    print(f"Institution token sent  : {_token_source()}\n")
    print("Sending a minimal Scopus Search API request...\n")

    resp = requests.get(SCOPUS_SEARCH_URL, headers=headers, params=params, timeout=30)

    print(f"  HTTP Status: {resp.status_code}")
    print()

    # Scopus returns quota info in response headers
    rate_headers = {
        "X-RateLimit-Limit": "Total allowed requests",
        "X-RateLimit-Remaining": "Requests remaining",
        "X-RateLimit-Reset": "Reset timestamp (Unix epoch)",
    }

    print("  --- Scopus Search API Quota ---")
    for header, description in rate_headers.items():
        value = resp.headers.get(header, "not provided")

        # Convert reset timestamp to human-readable
        if header == "X-RateLimit-Reset" and value != "not provided":
            from datetime import datetime, timezone
            try:
                reset_dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
                local_dt = reset_dt.astimezone()
                value = f"{value}  ({local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')})"
            except (ValueError, OSError):
                pass

        print(f"  {description:.<35s} {value}")

    print()

    # Also check if the response was successful
    if resp.status_code == 200:
        total = resp.json().get("search-results", {}).get("opensearch:totalResults", "?")
        print(f"  Test query returned {total} results — API key is working.")
    elif resp.status_code == 429:
        print("  WARNING: You have exceeded your rate limit!")
    else:
        if resp.status_code in (401, 403):
            print("  ERROR: Scopus refused the request. Its reason:")
        print(f"  {resp.text[:400]}")
        if resp.status_code in (401, 403):
            print(
                "\n  Common causes:\n"
                "   - 'Invalid API Key': the key is mistyped or incomplete, the .env edit\n"
                "     was not saved, or an old SCOPUS_API_KEY in the shell overrides .env.\n"
                "   - 'Requestor configuration settings insufficient' or an authorization\n"
                "     error: the key is valid but you are off your institution's network.\n"
                "     Connect to the institution's VPN, or add SCOPUS_API_INSTITUTION_KEY.\n"
                "   - 'Institution Token is not associated with API Key': a token is still\n"
                "     being sent (see 'Institution token sent' above). Empty both\n"
                "     SCOPUS_API_INSTITUTION_KEY and SCOPUS_INST_TOKEN, in .env and in the shell."
            )


if __name__ == "__main__":
    check_quota()
