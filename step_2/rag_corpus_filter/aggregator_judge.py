"""
Agent 4: Aggregator Judge.

Makes the final keep/discard decision for a paper using the paper profile,
condensed chunk summaries from Agents 2 & 3, deterministic metrics, and the
operational config.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_openai import ChatOpenAI

from rag_corpus_filter.config import PipelineConfig
from rag_corpus_filter.llm_utils import call_llm_structured, load_prompt
from rag_corpus_filter.models import (
    Chunk,
    ChunkRelevanceQuality,
    ChunkUtilityAssessment,
    DeterministicMetrics,
    OperationalConfig,
    PaperDecision,
    PaperProfile,
)

logger = logging.getLogger(__name__)


def _summarise_agent2(
    chunks: List[Chunk],
    results: List[ChunkRelevanceQuality],
) -> str:
    """Condense Agent 2 outputs into per-chunk summary lines.

    ACCS chunk weights (A3-w) are NOT surfaced here — they act only through
    the deterministic composite score and weighted criterion averages. Keeping
    the Judge prompt identical across A3 and A3-w makes the ablation a clean
    single-variable comparison (algorithmic weighting only).
    """
    lines: list[str] = []
    for chunk, r in zip(chunks, results):
        scores_str = ", ".join(
            f"{k}={v}" for k, v in r.criteria_scores.items()
        )
        flags = f" | anti-patterns: {r.anti_pattern_flags}" if r.anti_pattern_flags else ""
        snippets = ""
        if r.evidence_snippets:
            snippets = " | evidence: " + "; ".join(
                f'"{s.quote}"' for s in r.evidence_snippets[:2]
            )
        lines.append(
            f"{chunk.chunk_id} (pp.{chunk.start_page}-{chunk.end_page}): "
            f"relevance={r.domain_relevance_0_to_5}, labels=[{', '.join(r.assigned_labels)}], "
            f"scores=[{scores_str}], fig_dep={r.figure_table_dependency}, "
            f"standalone={r.standalone_comprehensibility}, "
            f"decision_rules={r.contains_decision_rules}, "
            f"quant_data={r.contains_quantitative_data}"
            f"{flags}{snippets}"
        )
    return "\n".join(lines)


def _summarise_agent3(
    chunks: List[Chunk],
    results: List[ChunkUtilityAssessment],
) -> str:
    """Condense Agent 3 outputs into per-chunk summary lines."""
    lines: list[str] = []
    for chunk, r in zip(chunks, results):
        query_scores = ", ".join(
            f"Q{i+1}={qa.utility_score_0_to_5}"
            for i, qa in enumerate(r.query_assessments)
        )
        best = ", ".join(r.best_served_queries[:3]) if r.best_served_queries else "none"
        lines.append(
            f"{chunk.chunk_id}: overall_utility={r.overall_utility_0_to_5}, "
            f"type={r.utility_type}, per_query=[{query_scores}], "
            f"best_served=[{best}]"
        )
    return "\n".join(lines)


def _format_metrics(metrics: DeterministicMetrics, include_utility: bool = True) -> str:
    """Format DeterministicMetrics for the prompt.

    Under the A4 ablation (include_utility=False) utility/query-coverage fields
    are omitted entirely so the Judge sees no trace of a subsystem it is not
    supposed to know about.
    """
    criteria_str = ", ".join(
        f"{k}={v:.2f}" for k, v in metrics.avg_criteria_scores.items()
    )
    lines = [
        f"Composite score: {metrics.composite_score_0_to_10}/10",
        f"Avg domain relevance: {metrics.avg_domain_relevance:.2f}/5",
        f"Avg criteria scores: {criteria_str}",
        f"Strong chunks: {metrics.strong_chunk_count}",
        f"Anti-pattern chunks: {metrics.anti_pattern_chunk_count}",
    ]
    if include_utility:
        coverage_detail = ", ".join(
            f"{q[:50]}={s}" for q, s in list(metrics.query_coverage_detail.items())[:10]
        )
        lines.extend([
            f"Avg utility: {metrics.avg_utility:.2f}/5",
            (
                f"Query coverage: {metrics.query_coverage:.1%} "
                f"({int(metrics.query_coverage * len(metrics.query_coverage_detail))}"
                f"/{len(metrics.query_coverage_detail)} queries served)"
            ),
            f"Direct-answer chunks: {metrics.direct_answer_chunk_count}",
            (
                f"Relevance-utility gap: {metrics.relevance_utility_gap:+.2f} "
                + ("(NOTE: gap may indicate raw data needing curation rather than low value)"
                   if metrics.relevance_utility_gap > 1.5 else "")
            ),
            f"Query coverage detail (best score per query): {coverage_detail}",
        ])
    return "\n".join(lines)


def _build_human_prompt(
    profile: PaperProfile,
    chunks: List[Chunk],
    agent2_results: List[ChunkRelevanceQuality],
    agent3_results: List[ChunkUtilityAssessment],
    metrics: DeterministicMetrics,
    config: OperationalConfig,
) -> str:
    """Assemble the human message for the aggregator judge."""
    parts: list[str] = []

    parts.append(f"APPLICATION CONTEXT:\n{config.application_description}")

    parts.append(
        f"PAPER PROFILE:\n"
        f"  Type: {profile.paper_type}\n"
        f"  Methodology: {profile.methodology_summary}\n"
        f"  Scope: {profile.scope}\n"
        f"  Data types: {', '.join(profile.data_types_present)}\n"
        f"  Domain fit: {profile.domain_fit_initial}\n"
        f"  Notes: {profile.profiler_notes}"
    )

    if agent3_results:
        parts.append(
            f"AGENT 2 — RELEVANCE & QUALITY ASSESSMENTS "
            f"({len(agent2_results)} chunks):\n"
            + _summarise_agent2(chunks, agent2_results)
        )
        parts.append(
            f"AGENT 3 — UTILITY SIMULATION "
            f"({len(agent3_results)} chunks):\n"
            + _summarise_agent3(chunks, agent3_results)
        )
    else:
        # A4 ablation: the Judge is shown only relevance/quality assessments,
        # with no reference (positive or negative) to any utility subsystem.
        parts.append(
            f"CHUNK-LEVEL RELEVANCE & QUALITY ASSESSMENTS "
            f"({len(agent2_results)} chunks):\n"
            + _summarise_agent2(chunks, agent2_results)
        )

    parts.append(
        "DETERMINISTIC METRICS:\n"
        + _format_metrics(metrics, include_utility=bool(agent3_results))
    )

    return "\n\n".join(parts)


def run_aggregator_judge(
    llm: ChatOpenAI,
    config: PipelineConfig,
    profile: PaperProfile,
    chunks: List[Chunk],
    agent2_results: List[ChunkRelevanceQuality],
    agent3_results: List[ChunkUtilityAssessment],
    metrics: DeterministicMetrics,
    operational_config: OperationalConfig,
    context_label: str = "aggregator_judge",
    paper_metrics=None,
) -> PaperDecision:
    """Run Agent 4 — Aggregator Judge.

    Takes condensed summaries (not raw chunk text) of all prior agent outputs
    plus deterministic metrics and produces the final paper decision.

    Under the A3-w condition, ACCS (Application-Conditioned Chunk Scoring)
    weights influence the Judge indirectly via the (already weighted)
    composite score and criterion averages in DeterministicMetrics. The
    per-chunk summary lines shown here are intentionally identical to those
    used under A3 so the Judge's qualitative reasoning sees the same raw
    evidence in both conditions.
    """
    # A4 ablation: use a utility-free system prompt so the Judge is never told
    # a utility subsystem exists. This keeps A4 a clean single-variable drop
    # of Agent 3 rather than "A3 minus Agent 3 with the Judge still asking for
    # it".
    prompt_file = (
        "aggregator_judge.txt" if agent3_results
        else "aggregator_judge_no_utility.txt"
    )
    system_prompt = load_prompt(config.paths.prompts_dir, prompt_file)
    human_prompt = _build_human_prompt(
        profile, chunks, agent2_results, agent3_results,
        metrics, operational_config,
    )

    logger.debug("Running aggregator judge for %s", context_label)
    return call_llm_structured(
        llm=llm,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        response_model=PaperDecision,
        config=config,
        context_label=context_label,
        paper_metrics=paper_metrics,
    )
