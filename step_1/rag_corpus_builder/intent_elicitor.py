"""
Phase 1 – Intent Elicitation Module

Conducts a multi-turn conversation with the user via an LLM to produce a
validated IntentBrief.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from pydantic import ValidationError

from rag_corpus_builder import config
from rag_corpus_builder.models import IntentBrief
from rag_corpus_builder.utils import extract_json_object, multiline_input

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_system_prompt() -> str:
    """Load the elicitation system prompt from file."""
    path = PROMPTS_DIR / "elicitation.txt"
    return path.read_text(encoding="utf-8")


def _validate_intent_brief(data: dict) -> IntentBrief | None:
    """Validate a dict against the IntentBrief schema."""
    try:
        return IntentBrief(**data)
    except ValidationError as exc:
        logger.warning("IntentBrief validation failed: %s", exc)
        return None


def run_elicitation(
    client: OpenAI, model: str,
) -> tuple[IntentBrief, List[Dict[str, str]]]:
    """Run the interactive intent elicitation dialogue.

    Returns (validated IntentBrief, conversation transcript).
    The transcript is the full messages list (excluding the system prompt).
    """
    system_prompt = _load_system_prompt()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # --- Get initial user description ---
    print("\n" + "=" * 60)
    print("  PHASE 1: Intent Elicitation")
    print("=" * 60)
    print(
        "\nDescribe the RAG corpus you want to build. Include as much "
        "detail as you can about the domain, purpose, and scope.\n"
    )
    user_input = multiline_input("You: ")
    if not user_input:
        user_input = "I need a corpus for my research."
    messages.append({"role": "user", "content": user_input})

    intent_brief: IntentBrief | None = None

    # --- Clarification loop (up to MAX_CLARIFICATION_ROUNDS) ---
    for round_num in range(1, config.MAX_CLARIFICATION_ROUNDS + 1):
        logger.info("Elicitation round %d / %d", round_num, config.MAX_CLARIFICATION_ROUNDS)

        # If this is the last round, nudge the LLM to generate the brief
        if round_num == config.MAX_CLARIFICATION_ROUNDS:
            messages.append({
                "role": "user",
                "content": (
                    "This is the final round. Based on everything discussed so far, "
                    "please generate the Intent Brief JSON now."
                ),
            })

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        assistant_msg = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": assistant_msg})

        # Check if the response contains a JSON intent brief
        extracted = extract_json_object(assistant_msg)
        if extracted is not None:
            intent_brief = _validate_intent_brief(extracted)
            if intent_brief is not None:
                break
            # JSON found but invalid — ask LLM to fix
            logger.info("JSON extracted but failed validation, requesting correction.")
            messages.append({
                "role": "user",
                "content": (
                    "The JSON you produced did not pass validation. "
                    "Please output a corrected JSON Intent Brief. "
                    "Required fields: domain, application_context, knowledge_axes "
                    "(list of {name, description}), scope_boundaries (list of strings), "
                    "content_anti_patterns (list of strings for chunk-level full-text evaluation), "
                    "depth_expectation, time_window. Optional: target_corpus_size (int or null)."
                ),
            })
            continue

        # LLM is asking clarification questions — show to user
        print(f"\nAssistant (round {round_num}):\n{assistant_msg}\n")

        if round_num < config.MAX_CLARIFICATION_ROUNDS:
            user_reply = multiline_input("You: ")
            if not user_reply:
                user_reply = "Please proceed with what you have."
            messages.append({"role": "user", "content": user_reply})

    # --- If loop ended without a valid brief, do one forced generation ---
    if intent_brief is None:
        logger.info("Forcing intent brief generation after dialogue.")
        messages.append({
            "role": "user",
            "content": (
                "Please generate the Intent Brief JSON now based on all information "
                "gathered. Output ONLY the JSON object."
            ),
        })
        for attempt in range(1 + config.LLM_JSON_RETRY_ATTEMPTS):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            assistant_msg = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": assistant_msg})

            extracted = extract_json_object(assistant_msg)
            if extracted is not None:
                intent_brief = _validate_intent_brief(extracted)
                if intent_brief is not None:
                    break

            if attempt < config.LLM_JSON_RETRY_ATTEMPTS:
                messages.append({
                    "role": "user",
                    "content": (
                        "That was not valid JSON matching the required schema. "
                        "Please try again. Output ONLY the raw JSON object."
                    ),
                })

    if intent_brief is None:
        raise RuntimeError(
            "Failed to produce a valid Intent Brief after all attempts. "
            "Please try again with a clearer description."
        )

    # --- User confirmation ---
    print("\n" + "-" * 60)
    print("  Generated Intent Brief")
    print("-" * 60)
    print(json.dumps(intent_brief.model_dump(), indent=2))
    print("-" * 60)

    while True:
        confirm = input("\nDo you confirm this intent brief? [y/n/edit]: ").strip().lower()
        if confirm in ("y", "yes"):
            logger.info("Intent brief confirmed by user.")
            transcript = [m for m in messages if m["role"] != "system"]
            return intent_brief, transcript
        elif confirm in ("n", "no"):
            # Let user provide corrections
            print("Please describe what should be changed:")
            edits = multiline_input("You: ")
            messages.append({
                "role": "user",
                "content": (
                    f"The user wants the following changes to the Intent Brief: {edits}\n"
                    "Please output a corrected JSON Intent Brief."
                ),
            })
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            assistant_msg = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": assistant_msg})

            extracted = extract_json_object(assistant_msg)
            if extracted is not None:
                validated = _validate_intent_brief(extracted)
                if validated is not None:
                    intent_brief = validated
                    print("\n" + "-" * 60)
                    print("  Updated Intent Brief")
                    print("-" * 60)
                    print(json.dumps(intent_brief.model_dump(), indent=2))
                    print("-" * 60)
                    continue
            print("Could not parse the updated brief. Keeping the previous version.")
        elif confirm in ("edit",):
            print("Please describe your edits:")
            edits = multiline_input("You: ")
            messages.append({
                "role": "user",
                "content": (
                    f"The user wants these edits: {edits}\n"
                    "Output the corrected JSON Intent Brief only."
                ),
            })
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            assistant_msg = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": assistant_msg})

            extracted = extract_json_object(assistant_msg)
            if extracted is not None:
                validated = _validate_intent_brief(extracted)
                if validated is not None:
                    intent_brief = validated
                    print("\n" + "-" * 60)
                    print("  Updated Intent Brief")
                    print("-" * 60)
                    print(json.dumps(intent_brief.model_dump(), indent=2))
                    print("-" * 60)
                    continue
            print("Could not parse the updated brief. Keeping the previous version.")
        else:
            print("Please enter 'y', 'n', or 'edit'.")
