"""
Agent 2: Relevance & Quality Assessor.

Evaluates each chunk for domain relevance and content quality against the
dynamically generated value criteria from the OperationalConfig.
One LLM call per chunk.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from rag_corpus_filter.config import PipelineConfig
from rag_corpus_filter.llm_utils import call_llm_structured, load_prompt
from rag_corpus_filter.models import ChunkRelevanceQuality, OperationalConfig

logger = logging.getLogger(__name__)


def _format_criteria_section(config: OperationalConfig) -> str:
    """Format value criteria with their scale anchors for prompt injection."""
    lines: list[str] = []
    for vc in config.value_criteria:
        lines.append(f"Criterion: {vc.name}")
        lines.append(f"  Description: {vc.description}")
        lines.append("  Scale anchors:")
        for score in sorted(vc.scale_anchors.keys()):
            lines.append(f"    {score}: {vc.scale_anchors[score]}")
        lines.append("")
    return "\n".join(lines)


def _build_human_prompt(chunk_text: str, config: OperationalConfig) -> str:
    """Assemble the human message with dynamic config injection."""
    parts: list[str] = []

    parts.append(
        f"APPLICATION CONTEXT:\n{config.application_description}"
    )

    parts.append(
        "DOMAIN TAXONOMY (assign labels from this list):\n"
        + ", ".join(config.domain_taxonomy)
    )

    parts.append(
        "VALUE CRITERIA (score each independently 0-5):\n"
        + _format_criteria_section(config)
    )

    parts.append(
        "ANTI-PATTERNS (red flags — if detected, list them and lower domain_relevance):\n"
        + "\n".join(f"  - {ap}" for ap in config.anti_patterns)
    )

    parts.append(f"CHUNK TEXT:\n{chunk_text}")

    return "\n\n".join(parts)


def run_relevance_assessor(
    llm: ChatOpenAI,
    config: PipelineConfig,
    chunk_text: str,
    operational_config: OperationalConfig,
    context_label: str = "relevance_assessor",
    paper_metrics=None,
) -> ChunkRelevanceQuality:
    """Run Agent 2 — Relevance & Quality Assessor on a single chunk.

    Parameters
    ----------
    chunk_text : The text content of the chunk to evaluate.
    operational_config : The generated operational config with criteria, taxonomy, etc.
    """
    system_prompt = load_prompt(config.paths.prompts_dir, "relevance_quality.txt")
    human_prompt = _build_human_prompt(chunk_text, operational_config)

    logger.debug("Running relevance assessor for %s", context_label)
    return call_llm_structured(
        llm=llm,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        response_model=ChunkRelevanceQuality,
        config=config,
        context_label=context_label,
        paper_metrics=paper_metrics,
    )
