"""
LLM client factory, JSON parsing with repair logic, retry wrapper.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type, TypeVar

if TYPE_CHECKING:
    from rag_corpus_filter.metrics_tracker import PaperMetrics

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from rag_corpus_filter.config import AgentModelMapping, LLMSettings, PipelineConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def build_llm_client(settings: LLMSettings, model_override: str | None = None) -> ChatOpenAI:
    """Create a ChatOpenAI instance pointed at LM Studio (or any OpenAI-compatible endpoint).

    Parameters
    ----------
    settings : LLMSettings — base endpoint/auth/temperature config.
    model_override : if provided, use this model ID instead of settings.model.
    """
    return ChatOpenAI(
        model=model_override or settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
    )


def build_agent_clients(
    settings: LLMSettings,
    mapping: AgentModelMapping,
) -> dict[str, ChatOpenAI]:
    """Build one ChatOpenAI client per agent role.

    Returns a dict with keys: 'agent1', 'agent2', 'agent3', 'agent4', 'config_prestep'.
    If a mapping field is None, falls back to settings.model (single-model mode).
    """
    roles = {
        "agent1": mapping.agent1_profiler,
        "agent2": mapping.agent2_relevance,
        "agent3": mapping.agent3_utility,
        "agent4": mapping.agent4_aggregator,
        "config_prestep": mapping.config_prestep,
    }
    clients = {}
    for role, model_id in roles.items():
        clients[role] = build_llm_client(settings, model_override=model_id)
    return clients


def validate_models_available(
    settings: LLMSettings,
    mapping: AgentModelMapping,
) -> list[str]:
    """Check that all configured models are available on the LLM server.

    Returns the list of available model IDs. For a local server, raises
    RuntimeError if a required model is missing. Cloud APIs do not always
    list every model ID they accept, so for those a missing model is only
    a warning (the first call fails loudly if the ID is wrong).
    """
    import openai as _openai

    from common.env import is_local_endpoint

    local = is_local_endpoint(settings.base_url)
    client = _openai.OpenAI(base_url=settings.base_url, api_key=settings.api_key)
    try:
        available = [m.id for m in client.models.list().data]
    except Exception as exc:
        if local:
            raise RuntimeError(f"Cannot reach the LLM server at {settings.base_url}: {exc}") from exc
        logger.warning("Could not list models at %s (%s); skipping the model check.", settings.base_url, exc)
        return []

    # Every role uses its own model if set, otherwise the default model.
    required = {
        model_id or settings.model
        for model_id in [
            mapping.agent1_profiler,
            mapping.agent2_relevance,
            mapping.agent3_utility,
            mapping.agent4_aggregator,
            mapping.config_prestep,
        ]
    }

    missing = required - set(available)
    if missing:
        msg = (
            f"Models not found on server: {sorted(missing)}\n"
            f"Available models: {available}"
        )
        if local:
            raise RuntimeError(msg + "\nPlease load the missing models in your local server first.")
        logger.warning(msg)
        return available

    logger.info("Model validation passed. Available: %s", available)
    return available


# ---------------------------------------------------------------------------
# JSON extraction / repair
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON."""
    text = text.strip()
    for marker in ("```json", "```JSON", "```"):
        if text.startswith(marker):
            text = text[len(marker):]
            break
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def find_json_object(text: str) -> str | None:
    """Find the outermost JSON object in a string by matching braces."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_safe(raw: str) -> dict | None:
    """Try multiple strategies to extract valid JSON from LLM output."""
    # Strategy 1: direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    cleaned = strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find JSON object in prose
    candidate = find_json_object(raw)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Bad output saving (for debugging)
# ---------------------------------------------------------------------------

def save_bad_output(raw: str, context: str, output_dir: Path) -> Path:
    """Save a raw LLM output that failed parsing, for debugging."""
    bad_dir = output_dir / "bad_outputs"
    bad_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", context)[:80]
    path = bad_dir / f"{slug}_{int(time.time())}.txt"
    path.write_text(raw, encoding="utf-8")
    logger.warning("Saved bad LLM output to %s", path)
    return path


# ---------------------------------------------------------------------------
# Raw OpenAI call with reasoning_content fallback
# ---------------------------------------------------------------------------

def _invoke_llm_raw(
    llm: ChatOpenAI,
    messages: list,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    """Call the LLM via the raw openai client to access reasoning_content.

    Some models (e.g. Gemma 4) use extended thinking and may place their
    output in ``reasoning_content`` instead of ``content``.  LangChain drops
    ``reasoning_content``, so we call the openai client directly.

    Setting ``max_tokens`` prevents thinking-mode models from exhausting
    the entire token budget on reasoning and leaving content empty.

    Returns (raw_text, source) where source is 'content' or 'reasoning'.
    """
    import openai as _openai

    client = _openai.OpenAI(
        base_url=str(llm.openai_api_base),
        api_key=llm.openai_api_key.get_secret_value() if hasattr(llm.openai_api_key, "get_secret_value") else (llm.openai_api_key or "lm-studio"),
    )

    # Convert LangChain messages to openai dicts
    api_msgs: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            api_msgs.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            api_msgs.append({"role": "user", "content": m.content})
        else:
            # AIMessage or similar — treat as assistant
            api_msgs.append({"role": "assistant", "content": getattr(m, "content", str(m))})

    resp = client.chat.completions.create(
        model=llm.model_name,
        messages=api_msgs,
        temperature=llm.temperature,
        max_tokens=max_tokens,
    )

    choice = resp.choices[0].message
    content = choice.content or ""
    if content.strip():
        return content, "content"

    # Fallback: check reasoning_content (extended thinking models)
    reasoning = getattr(choice, "reasoning_content", None) or ""
    if reasoning.strip():
        logger.info("Content was empty — extracting from reasoning_content")
        return reasoning, "reasoning"

    return "", "empty"


# ---------------------------------------------------------------------------
# Validated LLM call with retry + repair
# ---------------------------------------------------------------------------

def call_llm_structured(
    llm: ChatOpenAI,
    system_prompt: str,
    human_prompt: str,
    response_model: Type[T],
    config: PipelineConfig,
    context_label: str = "",
    paper_metrics: Optional["PaperMetrics"] = None,
) -> T:
    """Call the LLM, parse the JSON response, validate with Pydantic, retry on failure.

    Parameters
    ----------
    llm : ChatOpenAI
    system_prompt : str
    human_prompt : str
    response_model : Pydantic model class to validate against
    config : PipelineConfig (for retry settings and output dir)
    context_label : str — for logging and bad-output filenames

    Returns
    -------
    An instance of *response_model*.

    Raises
    ------
    RuntimeError if all retries are exhausted.
    """
    settings = config.llm
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    last_error: Exception | None = None

    for attempt in range(1, settings.max_retries + 1):
        # Track LLM call count and retries
        if paper_metrics is not None:
            from rag_corpus_filter.metrics_tracker import agent_tag_from_context_label
            tag = agent_tag_from_context_label(context_label)
            if tag != "other":
                setattr(paper_metrics, f"{tag}_llm_calls",
                        getattr(paper_metrics, f"{tag}_llm_calls") + 1)
                if attempt > 1:
                    setattr(paper_metrics, f"{tag}_retries",
                            getattr(paper_metrics, f"{tag}_retries") + 1)
        try:
            raw, source = _invoke_llm_raw(llm, messages, max_tokens=settings.max_tokens)
            logger.debug("[%s] attempt %d raw response (from %s):\n%s", context_label, attempt, source, raw[:500])

            parsed = parse_json_safe(raw)

            if parsed is None:
                save_bad_output(raw, f"{context_label}_parse_fail_a{attempt}", config.paths.output_dir)
                # Append assistant reply + repair request for next attempt
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=raw))
                messages.append(HumanMessage(content=(
                    "Your previous response was not valid JSON. "
                    "Please output ONLY the corrected JSON object, nothing else."
                )))
                continue

            # Validate with Pydantic
            try:
                return response_model.model_validate(parsed)
            except ValidationError as ve:
                save_bad_output(
                    json.dumps(parsed, indent=2),
                    f"{context_label}_validation_fail_a{attempt}",
                    config.paths.output_dir,
                )
                logger.warning("[%s] Pydantic validation failed: %s", context_label, ve)
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=raw))
                messages.append(HumanMessage(content=(
                    f"Your JSON was syntactically valid but failed schema validation:\n"
                    f"{ve}\n\n"
                    f"Please output a corrected JSON object that matches the required schema."
                )))
                last_error = ve
                continue

        except Exception as exc:
            logger.warning("[%s] LLM call error (attempt %d): %s", context_label, attempt, exc)
            last_error = exc
            if attempt < settings.max_retries:
                time.sleep(settings.retry_delay_sec)
            continue

    raise RuntimeError(
        f"[{context_label}] All {settings.max_retries} attempts exhausted. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Safe filename helper
# ---------------------------------------------------------------------------

def safe_filename(name: str) -> str:
    """Convert an arbitrary string to a filesystem-safe filename."""
    name = re.sub(r"[^\w\s\-.]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120]


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def load_prompt(prompts_dir: Path, filename: str) -> str:
    """Load a prompt text file from the prompts directory."""
    path = prompts_dir / filename
    return path.read_text(encoding="utf-8")
