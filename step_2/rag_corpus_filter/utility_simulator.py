"""
Agent 3: Utility Simulator.

Evaluates each chunk against ALL representative queries in a single LLM call,
simulating RAG retrieval to determine how useful the chunk would be as context
for answering each query.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from rag_corpus_filter.config import PipelineConfig
from rag_corpus_filter.llm_utils import call_llm_structured, load_prompt
from rag_corpus_filter.models import ChunkUtilityAssessment, OperationalConfig

logger = logging.getLogger(__name__)


def _build_human_prompt(chunk_text: str, config: OperationalConfig) -> str:
    """Assemble the human message with the chunk text and all queries."""
    parts: list[str] = []

    parts.append(
        f"APPLICATION CONTEXT:\n{config.application_description}"
    )

    query_lines = "\n".join(
        f"  Q{i+1}: {q}" for i, q in enumerate(config.representative_queries)
    )
    parts.append(
        f"REPRESENTATIVE QUERIES ({len(config.representative_queries)} total — "
        f"evaluate ALL of them):\n{query_lines}"
    )

    parts.append(f"CHUNK TEXT:\n{chunk_text}")

    return "\n\n".join(parts)


def run_utility_simulator(
    llm: ChatOpenAI,
    config: PipelineConfig,
    chunk_text: str,
    operational_config: OperationalConfig,
    context_label: str = "utility_simulator",
    paper_metrics=None,
) -> ChunkUtilityAssessment:
    """Run Agent 3 — Utility Simulator on a single chunk.

    Evaluates ALL representative queries against the chunk in one LLM call.

    Parameters
    ----------
    chunk_text : The text content of the chunk to evaluate.
    operational_config : The generated operational config with queries.
    """
    system_prompt = load_prompt(config.paths.prompts_dir, "utility_simulator.txt")
    human_prompt = _build_human_prompt(chunk_text, operational_config)

    logger.debug("Running utility simulator for %s", context_label)
    return call_llm_structured(
        llm=llm,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        response_model=ChunkUtilityAssessment,
        config=config,
        context_label=context_label,
        paper_metrics=paper_metrics,
    )
