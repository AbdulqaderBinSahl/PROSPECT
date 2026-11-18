"""
Configuration pre-step agent.

Takes an application description (+ optional knowledge axes and scope
boundaries from Step 1) and generates the OperationalConfig via a single
LLM call.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI

from rag_corpus_filter.config import PipelineConfig
from rag_corpus_filter.llm_utils import call_llm_structured, load_prompt
from rag_corpus_filter.models import OperationalConfig
from rag_corpus_filter.step1_loader import KnowledgeAxis

logger = logging.getLogger(__name__)


def _build_human_prompt(
    application_description: str,
    knowledge_axes: Optional[List[KnowledgeAxis]] = None,
    scope_boundaries: Optional[List[str]] = None,
    content_anti_patterns: Optional[List[str]] = None,
    depth_expectation: Optional[str] = None,
) -> str:
    """Assemble the human message for the config pre-step."""
    parts: List[str] = []

    parts.append(f"APPLICATION DESCRIPTION:\n{application_description}")

    if depth_expectation:
        parts.append(
            f"\nDEPTH EXPECTATION: {depth_expectation}\n"
            f"(Use this to calibrate criteria and thresholds — e.g., review "
            f"papers are more valuable for survey-level breadth, less so for "
            f"deep mechanistic detail.)"
        )

    if knowledge_axes:
        axes_text = "\n".join(
            f"  - {ax.name} [{ax.priority.upper()}]: {ax.description}"
            for ax in knowledge_axes
        )
        n_axes = len(knowledge_axes)
        if n_axes <= 3:
            scope_hint = "narrow, focused"
        elif n_axes <= 5:
            scope_hint = "moderate"
        else:
            scope_hint = "broad, multi-faceted"
        parts.append(
            f"\nKNOWLEDGE AXES ({n_axes} axes — this indicates a {scope_hint} "
            f"application scope; calibrate representative query count "
            f"accordingly. Priority labels indicate query weighting — "
            f"allocate at least 60% of queries to PRIMARY axes):\n{axes_text}"
        )

    if content_anti_patterns:
        # Prefer content_anti_patterns (chunk-level) over scope_boundaries
        # (search-level) when both are available.
        patterns_text = "\n".join(f"  - {p}" for p in content_anti_patterns)
        parts.append(
            f"\nCONTENT ANTI-PATTERNS (pre-defined for chunk-level evaluation — "
            f"use these directly as anti-patterns):\n{patterns_text}"
        )
    elif scope_boundaries:
        # Fallback for older Step 1 outputs that don't have content_anti_patterns
        boundaries_text = "\n".join(f"  - {b}" for b in scope_boundaries)
        parts.append(
            f"\nSCOPE BOUNDARIES (these are search-level exclusions — adapt "
            f"them into chunk-level anti-patterns, being careful not to over-"
            f"trigger on papers that mention excluded topics alongside "
            f"relevant content):\n{boundaries_text}"
        )

    return "\n\n".join(parts)


def run_config_prestep(
    llm: ChatOpenAI,
    config: PipelineConfig,
    application_description: str,
    knowledge_axes: Optional[List[KnowledgeAxis]] = None,
    scope_boundaries: Optional[List[str]] = None,
    content_anti_patterns: Optional[List[str]] = None,
    depth_expectation: Optional[str] = None,
) -> OperationalConfig:
    """Run the configuration pre-step agent.

    Returns a fully populated OperationalConfig.
    """
    system_prompt = load_prompt(config.paths.prompts_dir, "config_prestep.txt")
    human_prompt = _build_human_prompt(
        application_description, knowledge_axes, scope_boundaries,
        content_anti_patterns=content_anti_patterns,
        depth_expectation=depth_expectation,
    )

    logger.info("Running config pre-step agent...")
    result = call_llm_structured(
        llm=llm,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        response_model=OperationalConfig,
        config=config,
        context_label="config_prestep",
    )
    logger.info(
        "Config pre-step complete: %d queries, %d criteria, %d anti-patterns",
        len(result.representative_queries),
        len(result.value_criteria),
        len(result.anti_patterns),
    )
    return result
