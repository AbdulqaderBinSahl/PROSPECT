"""One small client for every model role in the evaluation.

A role (LLM judge, query paraphraser, reference drafter, RAG generator, gold
labeller, ...) is described in the settings file by a model spec:

    provider: openai | anthropic | google | local
    model: <model ID>
    temperature: <number>      # omitted = provider default (some models accept no other)
    max_tokens: <number>       # omitted = provider default (Anthropic requires one: 1024)
    json: true                 # ask for a JSON response (Google) and parse it (all)
    seed: 42                   # Google only
    timeout: 120               # seconds per call (Google; the SDK has none)
    retries: 3
    backoff: 10                # seconds x attempt between retries
    base_url: http://localhost:1234/v1   # provider "local": any OpenAI-compatible server
    cost_per_1m: [5.0, 30.0]   # optional USD per 1M input / output tokens, for budget guards

API keys are read from the environment / .env by name: OPENAI_API_KEY,
ANTHROPIC_API_KEY, GOOGLE_API_KEY (LOCAL_LLM_API_KEY for a local server that
needs one). They are never printed.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "local": "LOCAL_LLM_API_KEY",
}


@dataclass
class ModelSpec:
    provider: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    json: bool = False
    seed: Optional[int] = None
    timeout: Optional[float] = None
    retries: int = 3
    backoff: float = 10.0
    base_url: Optional[str] = None
    cost_per_1m: List[float] = field(default_factory=list)

    @classmethod
    def from_settings(cls, spec: Dict[str, Any]) -> "ModelSpec":
        known = set(cls.__dataclass_fields__)
        unknown = set(spec) - known - {"id", "label"}
        if unknown:
            raise ValueError(f"Unknown model setting(s) {sorted(unknown)} in {spec}")
        values = {k: v for k, v in spec.items() if k in known}
        if values.get("provider") not in KEY_ENV:
            raise ValueError(f"provider must be one of {sorted(KEY_ENV)}, got {spec.get('provider')!r}")
        return cls(**values)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def missing_key(spec: ModelSpec) -> Optional[str]:
    """Name of the API-key variable this spec needs but that is not set (None if fine)."""
    if spec.provider == "local":
        return None
    name = KEY_ENV[spec.provider]
    return None if os.environ.get(name) else name


class LLMClient:
    """Calls one model with retries. ``complete`` returns text; ``complete_json`` a dict."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        absent = missing_key(spec)
        if absent:
            raise SystemExit(f"\nERROR: {absent} is not set (needed for {spec.label}). Add it to .env.\n")
        key = os.environ.get(KEY_ENV[spec.provider], "")
        if spec.provider in ("openai", "local"):
            from openai import OpenAI

            if spec.provider == "local":
                self._client = OpenAI(base_url=spec.base_url or "http://localhost:1234/v1",
                                      api_key=key or "local")
            else:
                self._client = OpenAI(api_key=key)
        elif spec.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=key)
        else:
            from google import genai

            self._client = genai.Client(api_key=key)

    @property
    def model(self) -> str:
        return self.spec.model

    # ── public ──────────────────────────────────────────────────────────────

    def complete(self, prompt: str) -> Completion:
        last: Optional[Exception] = None
        for attempt in range(1, self.spec.retries + 1):
            try:
                result = self._call(prompt)
                if not result.text:
                    raise ValueError(f"empty response from {self.spec.label}")
                return result
            except Exception as exc:  # noqa: BLE001 — retry on any API error
                last = exc
                if attempt < self.spec.retries:
                    wait = self.spec.backoff * attempt
                    logger.warning("%s attempt %d/%d failed: %s — retrying in %.0fs",
                                   self.spec.label, attempt, self.spec.retries, exc, wait)
                    time.sleep(wait)
        raise RuntimeError(f"{self.spec.label} failed after {self.spec.retries} attempts") from last

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        """Call the model and parse a JSON object from its reply (retries parse failures too)."""
        last: Optional[Exception] = None
        for attempt in range(1, self.spec.retries + 1):
            text = self.complete(prompt).text
            try:
                return parse_json_object(text)
            except (ValueError, json.JSONDecodeError) as exc:
                last = exc
                logger.warning("%s returned unparseable JSON (attempt %d): %s",
                               self.spec.label, attempt, exc)
                if attempt < self.spec.retries:
                    time.sleep(self.spec.backoff * attempt)
        raise RuntimeError(f"{self.spec.label}: no valid JSON after {self.spec.retries} attempts") from last

    # ── providers ───────────────────────────────────────────────────────────

    def _cost(self, n_in: int, n_out: int) -> float:
        if len(self.spec.cost_per_1m) != 2:
            return 0.0
        return n_in / 1e6 * self.spec.cost_per_1m[0] + n_out / 1e6 * self.spec.cost_per_1m[1]

    def _call(self, prompt: str) -> Completion:
        s = self.spec
        if s.provider in ("openai", "local"):
            kwargs: Dict[str, Any] = {}
            if s.temperature is not None:
                kwargs["temperature"] = s.temperature
            if s.max_tokens is not None:
                kwargs["max_completion_tokens" if s.provider == "openai" else "max_tokens"] = s.max_tokens
            r = self._client.chat.completions.create(
                model=s.model, messages=[{"role": "user", "content": prompt}], **kwargs)
            n_in, n_out = r.usage.prompt_tokens, r.usage.completion_tokens
            return Completion(r.choices[0].message.content or "", n_in, n_out, self._cost(n_in, n_out))

        if s.provider == "anthropic":
            kwargs = {"max_tokens": s.max_tokens or 1024}
            if s.temperature is not None:
                kwargs["temperature"] = s.temperature
            r = self._client.messages.create(
                model=s.model, messages=[{"role": "user", "content": prompt}], **kwargs)
            text = r.content[0].text if r.content else ""
            n_in, n_out = r.usage.input_tokens, r.usage.output_tokens
            return Completion(text, n_in, n_out, self._cost(n_in, n_out))

        from google.genai import types

        cfg: Dict[str, Any] = {}
        if s.temperature is not None:
            cfg["temperature"] = s.temperature
        if s.json:
            cfg["response_mime_type"] = "application/json"
        if s.seed is not None:
            cfg["seed"] = s.seed
        if s.max_tokens is not None:
            cfg["max_output_tokens"] = s.max_tokens

        def call():
            return self._client.models.generate_content(
                model=s.model, contents=prompt, config=types.GenerateContentConfig(**cfg))

        if s.timeout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                r = pool.submit(call).result(timeout=s.timeout)
        else:
            r = call()
        meta = r.usage_metadata
        n_in = getattr(meta, "prompt_token_count", 0) or 0
        n_out = getattr(meta, "candidates_token_count", 0) or 0
        return Completion(r.text or "", n_in, n_out, self._cost(n_in, n_out))


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Parse the JSON object in a model reply, tolerating code fences and stray prose."""
    cleaned = _FENCE.sub("", raw.strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _OBJECT.search(cleaned)
        if not m:
            raise ValueError(f"no JSON object in reply: {raw[:200]!r}")
        data = json.loads(m.group())
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def client_for(settings_spec: Dict[str, Any]) -> LLMClient:
    return LLMClient(ModelSpec.from_settings(settings_spec))
