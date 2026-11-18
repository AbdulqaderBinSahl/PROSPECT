"""
Phase 2 – Search Strategy Generation Module

Takes a confirmed IntentBrief and uses the LLM to generate a Scopus-compatible
search strategy with concept groups, Boolean queries, and filters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from rag_corpus_builder import config
from rag_corpus_builder.models import IntentBrief, SearchStrategy
from rag_corpus_builder.utils import extract_json_object, multiline_input, llm_edit_query

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_system_prompt() -> str:
    path = PROMPTS_DIR / "strategy.txt"
    return path.read_text(encoding="utf-8")


def generate_strategy(
    client: OpenAI, model: str, intent_brief: IntentBrief
) -> SearchStrategy:
    """Generate a search strategy from the confirmed intent brief.

    Returns a validated SearchStrategy.
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: Search Strategy Generation")
    print("=" * 60)
    print("\nGenerating Scopus search strategy from intent brief...")

    system_prompt = _load_system_prompt()
    user_content = (
        "Here is the confirmed Intent Brief. Generate a complete Scopus search strategy.\n\n"
        f"{json.dumps(intent_brief.model_dump(), indent=2)}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    strategy: SearchStrategy | None = None

    for attempt in range(1 + config.LLM_JSON_RETRY_ATTEMPTS):
        logger.info(
            "Strategy generation attempt %d / %d",
            attempt + 1,
            1 + config.LLM_JSON_RETRY_ATTEMPTS,
        )

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
            try:
                strategy = SearchStrategy(**extracted)
                break
            except ValidationError as exc:
                logger.warning("SearchStrategy validation failed: %s", exc)
                messages.append({
                    "role": "user",
                    "content": (
                        f"The JSON did not pass schema validation:\n{exc}\n"
                        "Please fix the issues and output a corrected JSON object. "
                        "Remember: concept_groups needs 3-5 items, each with 3-8 terms."
                    ),
                })
        else:
            logger.warning("No JSON found in LLM response (attempt %d).", attempt + 1)
            messages.append({
                "role": "user",
                "content": (
                    "Your response did not contain valid JSON. "
                    "Please output ONLY the JSON object with no surrounding text."
                ),
            })

    if strategy is None:
        raise RuntimeError(
            "Failed to generate a valid search strategy. "
            "Check the LLM output or try a different model."
        )

    # Display summary and enter user review loop
    strategy = _user_review_strategy(client, model, strategy, intent_brief)

    logger.info("Search strategy confirmed by user.")
    return strategy


def _display_strategy(strategy: SearchStrategy) -> None:
    """Display the full strategy with complete query strings."""
    print(f"\n  Concept groups ({len(strategy.concept_groups)}):")
    for cg in strategy.concept_groups:
        print(f"    - {cg.name} ({len(cg.terms)} terms) -> axis: {cg.knowledge_axis}")

    print(f"\n  Queries ({len(strategy.queries)}):")
    print("-" * 60)
    for i, q in enumerate(strategy.queries, 1):
        print(f"\n  [{i}] Type: {q.query_type.upper()}")
        print(f"      Axes: {', '.join(q.knowledge_axes_served)}")
        print(f"      Groups: {', '.join(q.concept_groups_used)}")
        print("      Query:")
        print(f"      {q.query_string}")
    print("\n" + "-" * 60)

    print(f"\n  Time filter: {strategy.time_filter}")
    print(f"  Subject areas: {', '.join(strategy.subject_area_filters)}")


def _user_review_strategy(
    client: OpenAI, model: str, strategy: SearchStrategy, intent_brief: IntentBrief,
) -> SearchStrategy:
    """Interactive loop: let the user review, edit, add, or remove queries."""
    _display_strategy(strategy)

    while True:
        print("\n  Options:")
        print("    approve  — accept all queries and proceed to validation")
        print("    edit N   — describe changes, LLM rewrites query N")
        print("    manual N — paste a replacement for query N yourself")
        print("    remove N — remove query number N")
        print("    add      — add a new query manually")
        print("    show     — display all queries again")

        choice = input("\n  Action: ").strip().lower()

        if choice in ("approve", "a", "y", "yes"):
            if not strategy.queries:
                print("  Cannot proceed with zero queries. Add at least one.")
                continue
            print(f"\n  Approved {len(strategy.queries)} queries.")
            return strategy

        elif choice == "show":
            _display_strategy(strategy)

        elif choice.startswith("edit"):
            parts = choice.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("  Usage: edit N  (e.g. 'edit 3')")
                continue
            idx = int(parts[1])
            if idx < 1 or idx > len(strategy.queries):
                print(f"  Invalid query number. Choose 1-{len(strategy.queries)}.")
                continue

            q = strategy.queries[idx - 1]
            print(f"\n  Current query [{idx}]:")
            print(f"  {q.query_string}")
            print("\n  Describe what you want changed (e.g. 'add synonyms for impact strength'):")
            instruction = multiline_input("  Edit instruction: ")
            if not instruction:
                print(f"  No instruction — query [{idx}] unchanged.")
                continue

            print("  Asking LLM to rewrite...")
            new_qs = llm_edit_query(
                client, model, q.query_string, instruction,
                intent_brief.domain, intent_brief.application_context,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            if new_qs:
                print(f"\n  LLM proposed query [{idx}]:")
                print(f"  {new_qs}")
                confirm = input("\n  Accept this revision? [y/n]: ").strip().lower()
                if confirm in ("y", "yes"):
                    updated = q.model_copy(update={"query_string": new_qs})
                    strategy.queries[idx - 1] = updated
                    print(f"  Query [{idx}] updated.")
                else:
                    print(f"  Revision rejected — query [{idx}] unchanged.")
            else:
                print("  LLM failed to produce a revision. Try 'manual N' instead.")

        elif choice.startswith("manual"):
            parts = choice.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("  Usage: manual N  (e.g. 'manual 3')")
                continue
            idx = int(parts[1])
            if idx < 1 or idx > len(strategy.queries):
                print(f"  Invalid query number. Choose 1-{len(strategy.queries)}.")
                continue

            q = strategy.queries[idx - 1]
            print(f"\n  Current query [{idx}]:")
            print(f"  {q.query_string}")
            print("\n  Paste the replacement query (or press Enter to keep it):")
            new_qs = multiline_input("  New query: ")
            if new_qs:
                updated = q.model_copy(update={"query_string": new_qs})
                strategy.queries[idx - 1] = updated
                print(f"  Query [{idx}] updated.")
            else:
                print(f"  Query [{idx}] unchanged.")

        elif choice.startswith("remove"):
            parts = choice.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("  Usage: remove N  (e.g. 'remove 5')")
                continue
            idx = int(parts[1])
            if idx < 1 or idx > len(strategy.queries):
                print(f"  Invalid query number. Choose 1-{len(strategy.queries)}.")
                continue
            removed = strategy.queries.pop(idx - 1)
            print(f"  Removed query [{idx}]: {removed.query_string[:60]}...")

        elif choice == "add":
            print("\n  Enter the Scopus query string:")
            new_qs = multiline_input("  Query: ")
            if not new_qs:
                print("  Empty input — skipped.")
                continue
            qtype = input("  Query type (broad/narrow) [broad]: ").strip().lower() or "broad"
            axes_input = input("  Knowledge axes served (comma-separated): ").strip()
            axes = [a.strip() for a in axes_input.split(",") if a.strip()] if axes_input else []
            groups_input = input("  Concept groups used (comma-separated): ").strip()
            groups = [g.strip() for g in groups_input.split(",") if g.strip()] if groups_input else []

            from rag_corpus_builder.models import SearchQuery
            new_query = SearchQuery(
                query_string=new_qs,
                query_type=qtype,
                knowledge_axes_served=axes,
                concept_groups_used=groups,
            )
            strategy.queries.append(new_query)
            print(f"  Added query [{len(strategy.queries)}].")

        else:
            print("  Unknown action. Type 'approve', 'edit N', 'manual N', 'remove N', 'add', or 'show'.")
