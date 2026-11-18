"""
Phase 3 – Query Validation Module

Hybrid validation: hard safety caps catch extreme counts, then the LLM
judges every query in-between based on domain context.  Flagged queries
are sent back to the LLM for refinement (up to MAX_VALIDATION_ITERATIONS).
"""

from __future__ import annotations

import json
import logging
import time
from typing import List

import requests
from openai import OpenAI
from pydantic import ValidationError

from rag_corpus_builder import config
from rag_corpus_builder.models import (
    IntentBrief,
    SearchStrategy,
    SearchQuery,
    QueryValidationResult,
)
from rag_corpus_builder.utils import build_full_query, extract_json_array, multiline_input, llm_edit_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard safety caps — anything beyond these is flagged without asking the LLM
# Defaults are used when no target_corpus_size is provided.
# ---------------------------------------------------------------------------
DEFAULT_HARD_CAP_BROAD: int = 10_000
DEFAULT_HARD_CAP_NARROW: int = 5


def _compute_caps(target_corpus_size: int | None) -> tuple[int, int]:
    """Derive hard safety caps from the user's target corpus size.

    When a target is given, the broad cap is set to 5x the target (a single
    query returning more than 5x the total desired papers is almost certainly
    too diluted).  The narrow cap scales down proportionally but stays at
    least 3 to avoid being overly aggressive.
    """
    if target_corpus_size is None or target_corpus_size <= 0:
        return DEFAULT_HARD_CAP_BROAD, DEFAULT_HARD_CAP_NARROW

    broad = max(target_corpus_size * 5, 500)   # floor at 500
    narrow = max(target_corpus_size // 50, 3)   # floor at 3
    return broad, narrow


def _scopus_count(full_query: str) -> int:
    """Execute a count-only Scopus search and return total results."""
    headers = {
        "X-ELS-APIKey": config.SCOPUS_API_KEY,
        "Accept": "application/json",
    }
    if config.SCOPUS_INST_TOKEN:
        headers["X-ELS-Insttoken"] = config.SCOPUS_INST_TOKEN
    params = {
        "query": full_query,
        "count": 1,
        "start": 0,
    }

    logger.debug("Scopus count query: %s", full_query[:120])

    resp = requests.get(
        config.SCOPUS_BASE_URL, headers=headers, params=params, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()

    total = int(
        data.get("search-results", {}).get("opensearch:totalResults", 0)
    )
    return total


def _hard_classify(
    count: int,
    cap_broad: int = DEFAULT_HARD_CAP_BROAD,
    cap_narrow: int = DEFAULT_HARD_CAP_NARROW,
) -> str | None:
    """Return a status only if the count hits a hard safety cap.

    Returns None when the count is in the grey zone that needs LLM judgment.
    """
    if count > cap_broad:
        return "too_broad"
    if count < cap_narrow:
        return "too_narrow"
    return None  # needs LLM judgment


# ---------------------------------------------------------------------------
# LLM-based contextual judgment
# ---------------------------------------------------------------------------

def _llm_judge_queries(
    client: OpenAI,
    model: str,
    query_counts: List[dict],
    intent_brief: IntentBrief,
) -> List[dict]:
    """Ask the LLM to judge which queries need refinement based on context.

    Returns a list of dicts: {"query_string": ..., "verdict": "ok"|"too_broad"|"too_narrow", "reason": ...}
    """
    prompt = (
        "You are a Scopus search strategy expert. Given the research domain and "
        "the result counts for each query below, judge whether each query's result "
        "count is acceptable, too broad, or too narrow.\n\n"
        "Consider:\n"
        "- For niche/specific topics, even 20-50 results can be perfectly fine\n"
        "- For broad interdisciplinary topics, 2000-5000 results may be expected\n"
        "- A query is 'too_broad' if results would be unmanageable or too diluted\n"
        "- A query is 'too_narrow' if it misses important literature\n\n"
        f"Domain: {intent_brief.domain}\n"
        f"Application: {intent_brief.application_context}\n"
        f"Depth: {intent_brief.depth_expectation}\n\n"
        f"Queries and counts:\n{json.dumps(query_counts, indent=2)}\n\n"
        "Output ONLY a JSON array where each element has:\n"
        '  {"query_string": "...", "verdict": "ok" or "too_broad" or "too_narrow", '
        '"reason": "brief explanation"}\n'
        "Output ONLY the JSON array, no other text."
    )

    messages = [
        {"role": "system", "content": "You are a Scopus search validation expert."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        text = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM judgment call failed: %s", exc)
        return []

    # Parse JSON array
    parsed = extract_json_array(text)
    if parsed is None:
        logger.warning("Could not parse LLM judgment response.")
        return []

    return parsed


# ---------------------------------------------------------------------------
# LLM-based refinement
# ---------------------------------------------------------------------------

def _refine_queries_with_llm(
    client: OpenAI,
    model: str,
    flagged: List[QueryValidationResult],
    strategy: SearchStrategy,
    intent_brief: IntentBrief,
) -> List[SearchQuery]:
    """Ask the LLM to refine flagged queries."""
    issues = []
    for v in flagged:
        issues.append({
            "query": v.query_string,
            "count": v.result_count,
            "status": v.status,
        })

    prompt = (
        "Refine these flagged Scopus queries:\n\n"
        f"{json.dumps(issues, indent=2)}\n\n"
        f"Domain: {intent_brief.domain}. "
        "'too_broad' means too many results — make more specific. "
        "'too_narrow' means too few results — add synonyms or broaden terms.\n\n"
        'Output ONLY a JSON array: [{"query_string":"...","query_type":"broad/narrow",'
        '"knowledge_axes_served":[...],"concept_groups_used":[...]}]'
    )

    messages = [
        {"role": "system", "content": "You are a Scopus search refinement expert."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        text = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM call failed during query refinement: %s", exc)
        return []

    parsed = extract_json_array(text)
    if parsed is None or not isinstance(parsed, list):
        logger.warning("Could not parse refined queries from LLM.")
        return []

    refined: List[SearchQuery] = []
    for item in parsed:
        try:
            refined.append(SearchQuery(**item))
        except (ValidationError, TypeError) as exc:
            logger.warning("Skipping invalid refined query: %s", exc)

    return refined


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_validated_queries(
    current_queries: List[SearchQuery],
    results: List[QueryValidationResult],
) -> None:
    """Show every query with its full string, count, and status."""
    result_map = {vr.query_string: vr for vr in results}

    print("\n" + "-" * 60)
    print("  Validated Queries")
    print("-" * 60)

    for i, q in enumerate(current_queries, 1):
        vr = result_map.get(q.query_string)
        count = vr.result_count if vr else "?"
        status = vr.status if vr else "unknown"
        icon = {"ok": "OK", "too_broad": "BROAD", "too_narrow": "NARROW", "error": "ERR"}.get(status, "?")

        print(f"\n  [{i}] Status: {icon}  |  Results: {count}  |  Type: {q.query_type.upper()}")
        print(f"      Axes: {', '.join(q.knowledge_axes_served)}")
        print("      Query:")
        print(f"      {q.query_string}")

    print("\n" + "-" * 60)

    total = sum(vr.result_count for vr in results if vr.result_count > 0)
    ok = sum(1 for vr in results if vr.status == "ok")
    flagged = sum(1 for vr in results if vr.status in ("too_broad", "too_narrow"))
    print(f"  Total: {len(current_queries)} queries  |  {ok} ok, {flagged} flagged  |  ~{total} results")


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_queries(
    client: OpenAI,
    model: str,
    strategy: SearchStrategy,
    intent_brief: IntentBrief,
) -> tuple[SearchStrategy, List[QueryValidationResult]]:
    """Validate all queries using hybrid approach:
    1. Hard safety caps catch extremes (scaled to target_corpus_size)
    2. LLM judges everything in between based on domain context
    3. Flagged queries get refined by the LLM
    4. User reviews and approves before proceeding

    Returns the (possibly refined) strategy and the validation results.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: Query Validation (Hybrid: Safety Caps + LLM Judgment)")
    print("=" * 60)

    hard_cap_broad, hard_cap_narrow = _compute_caps(intent_brief.target_corpus_size)
    if intent_brief.target_corpus_size:
        print(f"\n  Target corpus size: {intent_brief.target_corpus_size} papers")
    print(f"  Safety caps: broad > {hard_cap_broad}, narrow < {hard_cap_narrow}")

    all_results: List[QueryValidationResult] = []
    current_queries = list(strategy.queries)

    for iteration in range(1, config.MAX_VALIDATION_ITERATIONS + 1):
        logger.info("Validation iteration %d / %d", iteration, config.MAX_VALIDATION_ITERATIONS)
        print(f"\n  Validation iteration {iteration}...")

        all_results = _run_one_validation_pass(
            client, model, current_queries, strategy, intent_brief,
            hard_cap_broad, hard_cap_narrow,
        )

        flagged = [vr for vr in all_results if vr.status in ("too_broad", "too_narrow")]
        ok_count = sum(1 for r in all_results if r.status == "ok")
        print(f"\n  Summary: {ok_count} ok, {len(flagged)} flagged, "
              f"{sum(1 for r in all_results if r.status == 'error')} errors")

        if not flagged:
            print("  All queries acceptable.")
            break

        if iteration < config.MAX_VALIDATION_ITERATIONS:
            print(f"\n  {len(flagged)} queries need refinement — asking LLM...")
            refined = _refine_queries_with_llm(
                client, model, flagged, strategy, intent_brief
            )
            if refined:
                flagged_strings = {v.query_string for v in flagged}
                kept = [q for q in current_queries if q.query_string not in flagged_strings]
                current_queries = kept + refined
                print(f"  Replaced {len(flagged)} queries with {len(refined)} refined queries.")
            else:
                print("  LLM did not return usable refinements; keeping original queries.")
                break
        else:
            print(f"\n  Max validation iterations reached. {len(flagged)} queries still flagged.")

    # -----------------------------------------------------------------------
    # User approval gate — the user must approve before retrieval starts
    # -----------------------------------------------------------------------
    _display_validated_queries(current_queries, all_results)

    current_queries, all_results = _user_approval_loop(
        client, model, current_queries, all_results, strategy, intent_brief,
        hard_cap_broad, hard_cap_narrow,
    )

    # Update strategy with approved queries
    refined_strategy = strategy.model_copy(update={"queries": current_queries})
    return refined_strategy, all_results


def _run_one_validation_pass(
    client: OpenAI,
    model: str,
    current_queries: List[SearchQuery],
    strategy: SearchStrategy,
    intent_brief: IntentBrief,
    hard_cap_broad: int,
    hard_cap_narrow: int,
) -> List[QueryValidationResult]:
    """Run counts + hard caps + LLM judgment for a list of queries.

    Returns a list of QueryValidationResult for every query.
    """
    # Step 1: Get Scopus counts
    count_map: dict[str, int] = {}
    for i, query in enumerate(current_queries):
        full_q = build_full_query(query, strategy)
        try:
            count = _scopus_count(full_q)
        except requests.RequestException as exc:
            logger.error("Scopus API error for query %d: %s", i + 1, exc)
            count = -1
        count_map[query.query_string] = count
        print(f"    Query {i+1}: {count:>6} results  —  {query.query_string[:60]}...")
        time.sleep(config.SCOPUS_REQUEST_DELAY)

    # Step 2: Hard safety caps
    hard_flagged: List[str] = []
    needs_llm_judgment: List[dict] = []
    for query in current_queries:
        count = count_map[query.query_string]
        if count < 0:
            continue
        hard_status = _hard_classify(count, hard_cap_broad, hard_cap_narrow)
        if hard_status is not None:
            hard_flagged.append(query.query_string)
            print(f"    [HARD CAP] {hard_status}: {count} results — {query.query_string[:50]}...")
        else:
            needs_llm_judgment.append({
                "query_string": query.query_string,
                "result_count": count,
                "query_type": query.query_type,
            })

    # Step 3: LLM judges grey-zone queries
    llm_verdicts: dict[str, str] = {}
    if needs_llm_judgment:
        print(f"\n  Asking LLM to judge {len(needs_llm_judgment)} queries in context...")
        judgments = _llm_judge_queries(client, model, needs_llm_judgment, intent_brief)
        for j in judgments:
            qs = j.get("query_string", "")
            verdict = j.get("verdict", "ok")
            reason = j.get("reason", "")
            llm_verdicts[qs] = verdict
            icon = {"ok": "+", "too_broad": "!", "too_narrow": "-"}.get(verdict, "?")
            print(f"    [{icon}] LLM: {verdict:>10}  —  {qs[:50]}...  ({reason})")

    # Step 4: Build results
    results: List[QueryValidationResult] = []
    for query in current_queries:
        count = count_map[query.query_string]
        if count < 0:
            status = "error"
        elif query.query_string in hard_flagged:
            hard_status = _hard_classify(count, hard_cap_broad, hard_cap_narrow)
            status = hard_status if hard_status else "ok"
        elif query.query_string in llm_verdicts:
            status = llm_verdicts[query.query_string]
        else:
            status = "ok"
        results.append(QueryValidationResult(
            query_string=query.query_string,
            result_count=count,
            status=status,
        ))

    return results


def _user_approval_loop(
    client: OpenAI,
    model: str,
    current_queries: List[SearchQuery],
    all_results: List[QueryValidationResult],
    strategy: SearchStrategy,
    intent_brief: IntentBrief,
    hard_cap_broad: int,
    hard_cap_narrow: int,
) -> tuple[List[SearchQuery], List[QueryValidationResult]]:
    """Let the user approve, edit, remove, or add queries before retrieval."""

    while True:
        print("\n  Options:")
        print("    approve   — accept queries and proceed to retrieval")
        print("    edit N    — describe changes, LLM rewrites query N")
        print("    manual N  — paste a replacement for query N yourself")
        print("    remove N  — remove query N")
        print("    add       — add a new query manually")
        print("    recheck   — re-run Scopus counts + LLM judgment on current queries")
        print("    show      — display all queries again")

        choice = input("\n  Action: ").strip().lower()

        if choice in ("approve", "a", "y", "yes"):
            if not current_queries:
                print("  Cannot proceed with zero queries. Add at least one.")
                continue
            print(f"\n  Approved {len(current_queries)} queries — proceeding to retrieval.")
            return current_queries, all_results

        elif choice == "show":
            _display_validated_queries(current_queries, all_results)

        elif choice.startswith("edit"):
            parts = choice.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("  Usage: edit N  (e.g. 'edit 3')")
                continue
            idx = int(parts[1])
            if idx < 1 or idx > len(current_queries):
                print(f"  Invalid query number. Choose 1-{len(current_queries)}.")
                continue

            q = current_queries[idx - 1]
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
                    old_qs = q.query_string
                    updated = q.model_copy(update={"query_string": new_qs})
                    current_queries[idx - 1] = updated
                    all_results = [vr for vr in all_results if vr.query_string != old_qs]
                    all_results.append(QueryValidationResult(
                        query_string=new_qs,
                        result_count=-1,
                        status="pending_recheck",
                    ))
                    print(f"  Query [{idx}] updated. Run 'recheck' to get its Scopus count.")
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
            if idx < 1 or idx > len(current_queries):
                print(f"  Invalid query number. Choose 1-{len(current_queries)}.")
                continue

            q = current_queries[idx - 1]
            print(f"\n  Current query [{idx}]:")
            print(f"  {q.query_string}")
            print("\n  Paste the replacement query (or press Enter to keep it):")
            new_qs = multiline_input("  New query: ")
            if new_qs:
                old_qs = q.query_string
                updated = q.model_copy(update={"query_string": new_qs})
                current_queries[idx - 1] = updated
                all_results = [vr for vr in all_results if vr.query_string != old_qs]
                all_results.append(QueryValidationResult(
                    query_string=new_qs,
                    result_count=-1,
                    status="pending_recheck",
                ))
                print(f"  Query [{idx}] updated. Run 'recheck' to get its Scopus count.")
            else:
                print(f"  Query [{idx}] unchanged.")

        elif choice.startswith("remove"):
            parts = choice.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("  Usage: remove N  (e.g. 'remove 5')")
                continue
            idx = int(parts[1])
            if idx < 1 or idx > len(current_queries):
                print(f"  Invalid query number. Choose 1-{len(current_queries)}.")
                continue
            removed = current_queries.pop(idx - 1)
            all_results = [vr for vr in all_results if vr.query_string != removed.query_string]
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

            new_query = SearchQuery(
                query_string=new_qs,
                query_type=qtype,
                knowledge_axes_served=axes,
                concept_groups_used=groups,
            )
            current_queries.append(new_query)
            all_results.append(QueryValidationResult(
                query_string=new_qs,
                result_count=-1,
                status="pending_recheck",
            ))
            print(f"  Added query [{len(current_queries)}]. Run 'recheck' to get its Scopus count.")

        elif choice == "recheck":
            print("\n  Re-running validation on all queries...")
            all_results = _run_one_validation_pass(
                client, model, current_queries, strategy, intent_brief,
                hard_cap_broad, hard_cap_narrow,
            )
            _display_validated_queries(current_queries, all_results)

        else:
            print("  Unknown action. Type 'approve', 'edit N', 'manual N', 'remove N', 'add', 'recheck', or 'show'.")
