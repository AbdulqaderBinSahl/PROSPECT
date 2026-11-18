"""
Shared utility functions for the RAG Corpus Builder pipeline.
"""

from __future__ import annotations

import json

from rag_corpus_builder.models import SearchQuery, SearchStrategy


# ---------------------------------------------------------------------------
# Multi-line input helper
# ---------------------------------------------------------------------------

def multiline_input(prompt: str = "You: ") -> str:
    """Read multi-line input from the user.

    Supports pasting long text.  Input ends when:
      - The user types END on its own line, or
      - The user presses Enter on an empty line (two consecutive newlines), or
      - EOF is received (Ctrl-D / Ctrl-Z).

    Returns the collected text with leading/trailing whitespace stripped.
    """
    print(prompt, end="", flush=True)
    print("  (paste or type your text; press Enter twice or type END to finish)")

    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            # Two consecutive empty lines → done
            if line == "" and lines and lines[-1] == "":
                lines.pop()  # remove the trailing blank
                break
            lines.append(line)
    except EOFError:
        pass

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> dict | None:
    """Try to extract a JSON object from LLM response text.

    Handles both raw JSON and JSON wrapped in markdown fences.
    """
    stripped = text.strip()

    # Try direct parse first (ideal case: LLM returns pure JSON)
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code fences
    for marker in ("```json", "```"):
        if marker in stripped:
            try:
                start = stripped.index(marker) + len(marker)
                end = stripped.index("```", start)
                candidate = stripped[start:end].strip()
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass

    return None


def extract_json_array(text: str) -> list | None:
    """Try to extract a JSON array from LLM response text."""
    stripped = text.strip()

    if stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    for marker in ("```json", "```"):
        if marker in stripped:
            try:
                start = stripped.index(marker) + len(marker)
                end = stripped.index("```", start)
                return json.loads(stripped[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    return None


# ---------------------------------------------------------------------------
# LLM-assisted query editing
# ---------------------------------------------------------------------------

def llm_edit_query(
    client,
    model: str,
    query_string: str,
    user_instruction: str,
    domain: str,
    application_context: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str | None:
    """Ask the LLM to rewrite a Scopus query based on a user instruction.

    Returns the revised query string, or None on failure.
    """
    import logging
    _logger = logging.getLogger(__name__)

    prompt = (
        "You are a Scopus search query expert.\n\n"
        f"Domain: {domain}\n"
        f"Application: {application_context}\n\n"
        f"Current query:\n{query_string}\n\n"
        f"User wants the following change:\n{user_instruction}\n\n"
        "Output ONLY the revised Scopus query string. No explanation, no JSON, "
        "no markdown fences — just the raw query string."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Scopus search query expert. Output only the revised query string."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content.strip()
        # Strip markdown fences if the LLM wraps it anyway
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            ).strip()
        return result if result else None
    except Exception as exc:
        _logger.error("LLM query edit failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Scopus query helpers
# ---------------------------------------------------------------------------

def build_full_query(query: SearchQuery, strategy: SearchStrategy) -> str:
    """Append time and subject-area filters to a base query string."""
    parts = [query.query_string]

    if strategy.time_filter:
        parts.append(f"AND {strategy.time_filter}")

    if strategy.subject_area_filters:
        subj = " OR ".join(
            f"SUBJAREA({code})" for code in strategy.subject_area_filters
        )
        parts.append(f"AND ({subj})")

    return " ".join(parts)
