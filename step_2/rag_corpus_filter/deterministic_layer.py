"""
Deterministic aggregation layer — pure computation, no LLM.

Implements composite scoring, strong-chunk counting, query coverage,
relevance-utility gap, and the evidence gate guardrail.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from rag_corpus_filter.models import (
    ChunkRelevanceQuality,
    ChunkUtilityAssessment,
    DeterministicMetrics,
    OperationalConfig,
    PaperDecision,
)

logger = logging.getLogger(__name__)


def compute_composite_score(
    agent2_results: List[ChunkRelevanceQuality],
    agent3_results: List[ChunkUtilityAssessment],
    config: OperationalConfig,
    chunk_weights: Optional[Dict[str, float]] = None,
    chunk_ids: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute the composite score (0-10).

    When chunk_weights is provided (A3-w / ACCS — Application-Conditioned
    Chunk Scoring), scores are weighted by application-conditioned chunk
    importance before averaging. When agent3_results is empty (A4 ablation),
    utility is excluded.

    Returns (composite_score, avg_criteria_scores_dict).
    """
    if not agent2_results:
        return 0.0, {}

    n = len(agent2_results)

    # Resolve per-chunk weights (default 1.0 = uniform)
    if chunk_weights and chunk_ids:
        w = [chunk_weights.get(cid, 1.0) for cid in chunk_ids]
    else:
        w = [1.0] * n
    w_sum = sum(w) or 1.0

    # Weighted average of each value criterion across chunks (from Agent 2)
    criterion_avgs: Dict[str, float] = {}
    for criterion in config.value_criteria:
        scores = [r.criteria_scores.get(criterion.name, 0) for r in agent2_results]
        criterion_avgs[criterion.name] = sum(s * wi for s, wi in zip(scores, w)) / w_sum

    all_avgs = list(criterion_avgs.values())

    # Include utility if Agent 3 data is available
    if agent3_results:
        utility_scores = [r.overall_utility_0_to_5 for r in agent3_results]
        utility_avg = sum(s * wi for s, wi in zip(utility_scores, w)) / w_sum
        all_avgs.append(utility_avg)

    # All dimensions weighted equally, normalize to 0-10
    combined = sum(all_avgs) / len(all_avgs)  # 0-5 scale
    composite = round(combined * 2.0, 2)  # scale to 0-10

    return composite, criterion_avgs


def count_strong_chunks(
    agent2_results: List[ChunkRelevanceQuality],
    config: OperationalConfig,
) -> int:
    """Count chunks where enough criteria meet the score threshold."""
    count = 0
    for chunk_result in agent2_results:
        criteria_met = sum(
            1 for score in chunk_result.criteria_scores.values()
            if score >= config.strong_chunk_score_threshold
        )
        if criteria_met >= config.strong_chunk_min_criteria:
            count += 1
    return count


def compute_query_coverage(
    agent3_results: List[ChunkUtilityAssessment],
    config: OperationalConfig,
) -> Tuple[float, Dict[str, int]]:
    """Compute fraction of representative queries served by at least one chunk.

    A query is "served" if any chunk scores utility >= 3 for that query.

    Returns (coverage_fraction, per_query_best_scores).
    """
    if not agent3_results or not config.representative_queries:
        return 0.0, {}

    # Per query: best utility score across all chunks
    best_per_query: Dict[str, int] = {q: 0 for q in config.representative_queries}

    for chunk_assessment in agent3_results:
        for qa in chunk_assessment.query_assessments:
            if qa.query in best_per_query:
                best_per_query[qa.query] = max(best_per_query[qa.query], qa.utility_score_0_to_5)

    served = sum(1 for score in best_per_query.values() if score >= 3)
    coverage = served / len(config.representative_queries)

    return round(coverage, 4), best_per_query


def compute_relevance_utility_gap(
    agent2_results: List[ChunkRelevanceQuality],
    agent3_results: List[ChunkUtilityAssessment],
) -> float:
    """Compute avg_domain_relevance - avg_utility.

    Large positive value = relevant but not useful (RED FLAG).
    """
    if not agent2_results or not agent3_results:
        return 0.0

    avg_relevance = sum(r.domain_relevance_0_to_5 for r in agent2_results) / len(agent2_results)
    avg_utility = sum(r.overall_utility_0_to_5 for r in agent3_results) / len(agent3_results)
    return round(avg_relevance - avg_utility, 4)


def build_deterministic_metrics(
    agent2_results: List[ChunkRelevanceQuality],
    agent3_results: List[ChunkUtilityAssessment],
    config: OperationalConfig,
    chunk_weights: Optional[Dict[str, float]] = None,
    chunk_ids: Optional[List[str]] = None,
) -> DeterministicMetrics:
    """Build the full DeterministicMetrics object from agent outputs.

    When agent3_results is empty (A4 ablation), utility-related fields
    are set to 0.0 and coverage detail is empty.

    When chunk_weights is provided (A3-w / ACCS condition), scores are
    weighted by the application-conditioned chunk importance before averaging.
    chunk_ids must also be provided to map weights to results.
    """
    composite, avg_criteria = compute_composite_score(
        agent2_results, agent3_results, config,
        chunk_weights=chunk_weights, chunk_ids=chunk_ids,
    )
    strong_chunks = count_strong_chunks(agent2_results, config)
    coverage, coverage_detail = compute_query_coverage(agent3_results, config)
    gap = compute_relevance_utility_gap(agent2_results, agent3_results)

    avg_relevance = (
        sum(r.domain_relevance_0_to_5 for r in agent2_results) / len(agent2_results)
        if agent2_results else 0.0
    )
    avg_utility = (
        sum(r.overall_utility_0_to_5 for r in agent3_results) / len(agent3_results)
        if agent3_results else 0.0
    )
    anti_pattern_count = sum(
        1 for r in agent2_results if r.anti_pattern_flags
    )
    direct_answer_count = sum(
        1 for r in agent3_results if r.utility_type == "direct_answer"
    )

    return DeterministicMetrics(
        avg_domain_relevance=round(avg_relevance, 4),
        avg_criteria_scores={k: round(v, 4) for k, v in avg_criteria.items()},
        strong_chunk_count=strong_chunks,
        anti_pattern_chunk_count=anti_pattern_count,
        avg_utility=round(avg_utility, 4),
        query_coverage=coverage,
        query_coverage_detail=coverage_detail,
        direct_answer_chunk_count=direct_answer_count,
        relevance_utility_gap=gap,
        composite_score_0_to_10=composite,
    )


def apply_evidence_gate(
    decision: PaperDecision,
    strong_chunks: int,
    config: OperationalConfig,
) -> Tuple[str, str]:
    """Post Agent 4 evidence gate — log-only, does not override.

    Flags cases where Agent 4 says 'keep' but strong_chunk_count is below
    threshold.  The flag is recorded for analysis but the LLM decision stands.

    Returns (final_decision, evidence_gate_note).
    """
    if decision.decision == "keep" and strong_chunks < config.min_strong_chunks_for_keep:
        note = (
            f"evidence_gate_flag: strong_chunks={strong_chunks} "
            f"< {config.min_strong_chunks_for_keep}"
        )
        logger.info("Evidence gate flagged (no override): %s", note)
        return decision.decision, note
    return decision.decision, ""
