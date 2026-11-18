"""
Agent 1: Paper Profiler.

Produces a factual characterization of a paper from its abstract (Step 1 JSON
or first-pages fallback), last pages (conclusion), and metadata. No evaluation.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_openai import ChatOpenAI

from rag_corpus_filter.config import PipelineConfig
from rag_corpus_filter.llm_utils import call_llm_structured, load_prompt
from rag_corpus_filter.models import Chunk, ChunkWeightsResponse, OperationalConfig, PaperProfile
from rag_corpus_filter.step1_loader import PaperRecord

_NUMERIC_TOKEN = re.compile(r"\b\d+(?:\.\d+)?\b")


def _chunk_position(idx: int, total: int) -> str:
    """Rough position of a chunk within the paper."""
    if total <= 1:
        return "first"
    third = total / 3
    if idx < third:
        return "first"
    if idx < 2 * third:
        return "middle"
    return "last"


def _chunk_preview(text: str, head_words: int = 200, tail_words: int = 50) -> str:
    """First head_words + last tail_words of a chunk (to catch section transitions)."""
    words = text.split()
    if len(words) <= head_words + tail_words:
        return " ".join(words)
    head = " ".join(words[:head_words])
    tail = " ".join(words[-tail_words:])
    return f"{head} ... {tail}"

logger = logging.getLogger(__name__)


def _build_human_prompt(
    abstract_text: str,
    last_pages_text: str,
    domain: str,
    metadata: Optional[PaperRecord] = None,
) -> str:
    """Assemble the human message for the paper profiler."""
    parts: list[str] = []

    parts.append(f"APPLICATION DOMAIN:\n{domain}")

    if metadata:
        meta_lines = [f"Title: {metadata.title}"]
        if metadata.document_type:
            meta_lines.append(f"Document type (Scopus): {metadata.document_type}")
        if metadata.source_title:
            meta_lines.append(f"Journal: {metadata.source_title}")
        if metadata.publication_year:
            meta_lines.append(f"Year: {metadata.publication_year}")
        if metadata.author_keywords:
            meta_lines.append(f"Author keywords: {metadata.author_keywords}")
        if metadata.indexed_keywords:
            meta_lines.append(f"Indexed keywords: {metadata.indexed_keywords}")
        if metadata.retrieved_by_queries:
            meta_lines.append(
                f"Matched {len(metadata.retrieved_by_queries)} search queries "
                f"(higher = more central to domain)"
            )
        parts.append("METADATA:\n" + "\n".join(meta_lines))

    parts.append(f"ABSTRACT:\n{abstract_text}")
    parts.append(f"CONCLUSION / LAST PAGES TEXT:\n{last_pages_text}")

    return "\n\n".join(parts)


def format_paper_context(
    abstract_text: str,
    last_pages_text: str,
    domain: str,
    metadata: Optional[PaperRecord] = None,
) -> str:
    """The paper context exactly as Agent 1 receives it.

    The A2 baseline embeds this same block in its per-chunk prompt, so A2 and
    the multi-agent conditions profile each paper from identical inputs.
    """
    return _build_human_prompt(abstract_text, last_pages_text, domain, metadata)


def run_paper_profiler(
    llm: ChatOpenAI,
    config: PipelineConfig,
    abstract_text: str,
    last_pages_text: str,
    domain: str,
    metadata: Optional[PaperRecord] = None,
    context_label: str = "paper_profiler",
    paper_metrics=None,
) -> PaperProfile:
    """Run Agent 1 — Paper Profiler.

    Parameters
    ----------
    abstract_text : The abstract (from Step 1 JSON, or first-pages fallback).
    last_pages_text : Raw text from the last N pages of the PDF.
    domain : Application domain string (for domain_fit_initial).
    metadata : Optional Scopus metadata from Step 1.
    """
    system_prompt = load_prompt(config.paths.prompts_dir, "paper_profiler.txt")
    human_prompt = _build_human_prompt(
        abstract_text, last_pages_text, domain, metadata
    )

    logger.debug("Running paper profiler for %s", context_label)
    return call_llm_structured(
        llm=llm,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        response_model=PaperProfile,
        config=config,
        context_label=context_label,
        paper_metrics=paper_metrics,
    )


def _build_chunk_weights_prompt(
    profile: PaperProfile,
    chunks: list[Chunk],
    operational_config: OperationalConfig,
) -> str:
    """Assemble the human message for per-chunk weight assignment."""
    parts: list[str] = []

    parts.append(f"APPLICATION DESCRIPTION:\n{operational_config.application_description}")

    parts.append(
        f"PAPER PROFILE:\n"
        f"  Type: {profile.paper_type}\n"
        f"  Methodology: {profile.methodology_summary}\n"
        f"  Scope: {profile.scope}\n"
        f"  Data types: {', '.join(profile.data_types_present)}\n"
        f"  Domain fit: {profile.domain_fit_initial}"
    )

    # Show first ~200 + last ~50 words of each chunk, with structural signals
    # (position in paper, numeric density as a proxy for tables/results).
    total = len(chunks)
    chunk_previews = []
    for i, chunk in enumerate(chunks):
        preview = _chunk_preview(chunk.text)
        position = _chunk_position(i, total)
        numeric_count = len(_NUMERIC_TOKEN.findall(chunk.text))
        numeric_flag = "high" if numeric_count > 30 else "medium" if numeric_count > 10 else "low"
        chunk_previews.append(
            f"  {chunk.chunk_id} (pp.{chunk.start_page}-{chunk.end_page}, "
            f"{chunk.word_count} words, position={position}, "
            f"numeric_density={numeric_flag} [{numeric_count} numbers]):\n"
            f"    {preview}"
        )
    parts.append(f"CHUNKS ({len(chunks)} total):\n" + "\n\n".join(chunk_previews))

    return "\n\n".join(parts)


def run_chunk_weighting(
    llm: ChatOpenAI,
    config: PipelineConfig,
    profile: PaperProfile,
    chunks: list[Chunk],
    operational_config: OperationalConfig,
    context_label: str = "chunk_weights",
    paper_metrics=None,
) -> ChunkWeightsResponse:
    """Run the ACCS weighting step (Application-Conditioned Chunk Scoring, A3-w condition).

    Uses Agent 1's LLM client to assign application-conditioned weights
    to each chunk based on the paper profile and chunk previews.

    Returns the full ``ChunkWeightsResponse`` (chunk_id, section_type,
    weight, reasoning per chunk). Any chunk missing from the LLM's response
    is backfilled with weight 1.0 and section_type ``"other"`` so that
    downstream code can treat the result as complete. Use ``.chunk_weights``
    to iterate the entries, or ``weights_as_dict(response)`` below to extract
    the ``chunk_id -> weight`` mapping the deterministic layer expects.
    """
    from rag_corpus_filter.models import ChunkWeight  # local to avoid cycles

    system_prompt = load_prompt(config.paths.prompts_dir, "chunk_weights.txt")
    human_prompt = _build_chunk_weights_prompt(profile, chunks, operational_config)

    logger.debug("Running chunk weighting for %s", context_label)
    result = call_llm_structured(
        llm=llm,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        response_model=ChunkWeightsResponse,
        config=config,
        context_label=context_label,
        paper_metrics=paper_metrics,
    )

    # Backfill any missing chunks with a neutral default (weight=1.0, "other")
    present = {cw.chunk_id for cw in result.chunk_weights}
    for chunk in chunks:
        if chunk.chunk_id not in present:
            logger.warning(
                "[%s] Chunk %s missing from weights response, defaulting to 1.0/other",
                context_label, chunk.chunk_id,
            )
            result.chunk_weights.append(ChunkWeight(
                chunk_id=chunk.chunk_id,
                section_type="other",
                weight=1.0,
                reasoning="backfilled: chunk missing from LLM response",
            ))

    # Preserve chunk ordering to match the input list (useful for JSONL join).
    order = {c.chunk_id: i for i, c in enumerate(chunks)}
    result.chunk_weights.sort(key=lambda cw: order.get(cw.chunk_id, len(order)))

    logger.info(
        "[%s] Chunk weights: %s",
        context_label,
        {cw.chunk_id: f"{cw.weight:.2f}" for cw in result.chunk_weights},
    )
    return result


def weights_as_dict(response: ChunkWeightsResponse) -> dict[str, float]:
    """Extract the ``chunk_id -> weight`` mapping used by the deterministic layer."""
    return {cw.chunk_id: cw.weight for cw in response.chunk_weights}
