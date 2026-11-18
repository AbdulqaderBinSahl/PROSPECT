"""Environment loading and LLM endpoint resolution shared by Steps 1 and 2.

PROSPECT talks to any OpenAI-compatible chat endpoint: a local server
(LM Studio, Ollama, vLLM) or a cloud API (OpenAI, or any provider that
offers an OpenAI-compatible endpoint). The endpoint is resolved in this
order: command-line flag > environment variable / .env > default.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_URL = "http://localhost:1234/v1"  # LM Studio default
DEFAULT_LOCAL_API_KEY = "lm-studio"  # local servers ignore the key
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def load_env() -> None:
    """Load ``<repo>/.env`` into ``os.environ``.

    Variables already set in the shell take precedence over the file.
    Must be called before importing modules that read the environment at
    import time (e.g. ``rag_corpus_builder.config``).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at runtime
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def is_local_endpoint(base_url: str) -> bool:
    """True if the endpoint points at this machine (local LLM server)."""
    host = (urlparse(base_url).hostname or "").lower()
    return host in _LOCAL_HOSTS


@dataclass
class LLMEndpoint:
    """Resolved connection settings for the chat LLM."""
    base_url: str
    api_key: str
    model: str

    @property
    def is_local(self) -> bool:
        return is_local_endpoint(self.base_url)


def _detect_local_model(base_url: str, api_key: str) -> str:
    """Return the first model listed by a local server (LM Studio-style)."""
    from openai import OpenAI

    try:
        listed = [m.id for m in OpenAI(base_url=base_url, api_key=api_key).models.list().data]
    except Exception as exc:  # server down, wrong port, ...
        sys.exit(
            f"\nERROR: cannot reach the LLM server at {base_url} ({exc}).\n"
            "Start your local server (e.g. LM Studio) or set LLM_BASE_URL in .env.\n"
        )
    if not listed:
        sys.exit(f"\nERROR: no models are loaded on {base_url}. Load a model first.\n")
    if len(listed) > 1:
        logger.warning(
            "Several models are available on %s; using the first (%s). "
            "Set LLM_MODEL in .env to choose explicitly.", base_url, listed[0],
        )
    return listed[0]


def resolve_llm_endpoint(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> LLMEndpoint:
    """Resolve the LLM endpoint from CLI values, then ``LLM_*`` env vars, then defaults.

    For a remote API both ``LLM_API_KEY`` and ``LLM_MODEL`` are required.
    For a local server the model is auto-detected when not given.
    """
    base_url = base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL
    api_key = api_key or os.getenv("LLM_API_KEY") or ""
    model = model or os.getenv("LLM_MODEL") or ""

    if is_local_endpoint(base_url):
        api_key = api_key or DEFAULT_LOCAL_API_KEY
        if not model:
            model = _detect_local_model(base_url, api_key)
    else:
        missing = [name for name, val in (("LLM_API_KEY", api_key), ("LLM_MODEL", model)) if not val]
        if missing:
            sys.exit(
                f"\nERROR: {' and '.join(missing)} must be set to use the remote endpoint "
                f"{base_url}.\nAdd them to .env (see .env.example).\n"
            )

    return LLMEndpoint(base_url=base_url, api_key=api_key, model=model)


def scopus_keys() -> tuple[str, str]:
    """Return (api_key, institution_token) for the Elsevier Scopus APIs.

    ``SCOPUS_API_INSTITUTION_KEY`` is the documented name; the older
    ``SCOPUS_INST_TOKEN`` is still accepted.
    """
    api_key = os.getenv("SCOPUS_API_KEY", "")
    inst = os.getenv("SCOPUS_API_INSTITUTION_KEY") or os.getenv("SCOPUS_INST_TOKEN", "")
    return api_key, inst
