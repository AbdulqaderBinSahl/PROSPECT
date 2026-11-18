"""
Output generation: Excel workbook, JSONL chunk log, per-paper JSON,
PDF routing, and operational config saving.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from rag_corpus_filter.config import PathSettings
from rag_corpus_filter.models import (
    Chunk,
    ChunkRelevanceQuality,
    ChunkUtilityAssessment,
    ChunkWeight,
    DeterministicMetrics,
    OperationalConfig,
    PaperDecision,
    PaperProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-paper outputs
# ---------------------------------------------------------------------------

def save_paper_decision(
    pdf_name: str,
    decision: PaperDecision,
    output_dir: Path,
    profile: Optional[PaperProfile] = None,
    metrics: Optional[DeterministicMetrics] = None,
    paper_metrics: Optional[Dict[str, Any]] = None,
    chunk_weights_detail: Optional[List[ChunkWeight]] = None,
    application_description: Optional[str] = None,
) -> Path:
    """Save full PaperDecision as JSON in paper_decisions/.

    Optionally embeds the Agent 1 profile, deterministic metrics, run
    metrics, and (under A3-w / ACCS) the per-chunk weights alongside the
    Agent 4 decision for full traceability and post-hoc audit. When
    ``chunk_weights_detail`` AND ``application_description`` are both
    provided, the (truncated) application description is also embedded so
    the decision JSON is a self-contained audit artefact — a reviewer does
    not need to cross-reference operational_config.json to see what the
    ACCS weights were conditioned on.
    """
    decisions_dir = output_dir / "paper_decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(pdf_name).stem
    path = decisions_dir / f"{stem}_{ts}.json"

    record: Dict[str, Any] = decision.model_dump()
    if profile is not None:
        record["agent1_profile"] = profile.model_dump()
    if metrics is not None:
        record["deterministic_metrics"] = metrics.model_dump()
    if paper_metrics is not None:
        record["run_metrics"] = paper_metrics
    if chunk_weights_detail is not None:
        record["chunk_weights"] = [cw.model_dump() for cw in chunk_weights_detail]
        if application_description:
            record["application_description"] = application_description[:2000]

    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved decision for %s -> %s", pdf_name, path.name)
    return path


def route_pdf(
    pdf_path: Path,
    final_decision: str,
    paths: PathSettings,
) -> Path:
    """Copy PDF to the appropriate routed_pdfs/ subfolder."""
    dest_dir = {
        "keep": paths.keep_dir,
        "discard": paths.discard_dir,
    }.get(final_decision, paths.discard_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / pdf_path.name
    shutil.copy2(str(pdf_path), str(dest))
    logger.info("Routed %s -> %s/", pdf_path.name, final_decision)
    return dest


# ---------------------------------------------------------------------------
# JSONL chunk log
# ---------------------------------------------------------------------------

def write_chunk_jsonl(
    pdf_name: str,
    chunks: List[Chunk],
    agent2_results: List[ChunkRelevanceQuality],
    agent3_results: List[ChunkUtilityAssessment],
    jsonl_path: Path,
    chunk_weights_detail: Optional[List[ChunkWeight]] = None,
) -> None:
    """Append one JSON line per chunk (Agent 2 + Agent 3 outputs).

    When agent3_results is empty (A4 ablation), the agent3 field is null.
    Under A3-w the adaptive weighting detail (section_type, weight,
    reasoning) is embedded under ``chunk_weight`` for downstream audit;
    it is null for all other conditions.
    """
    weight_by_id: Dict[str, ChunkWeight] = (
        {cw.chunk_id: cw for cw in chunk_weights_detail}
        if chunk_weights_detail else {}
    )

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for i, (chunk, a2) in enumerate(zip(chunks, agent2_results)):
            a3 = agent3_results[i] if i < len(agent3_results) else None
            cw = weight_by_id.get(chunk.chunk_id)
            record = {
                "pdf": pdf_name,
                "chunk_id": chunk.chunk_id,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "word_count": chunk.word_count,
                "agent2": a2.model_dump(),
                "agent3": a3.model_dump() if a3 is not None else None,
                "chunk_weight": cw.model_dump() if cw is not None else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Operational config saving
# ---------------------------------------------------------------------------

def save_operational_config(config: OperationalConfig, output_dir: Path) -> Path:
    """Save OperationalConfig as JSON for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "operational_config.json"
    path.write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("Saved operational config -> %s", path)
    return path


# ---------------------------------------------------------------------------
# Excel workbook (3 sheets)
# ---------------------------------------------------------------------------

def _build_papers_df(paper_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build the Papers sheet DataFrame."""
    return pd.DataFrame(paper_rows)


def _build_chunks_df(chunk_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build the Chunks sheet DataFrame."""
    return pd.DataFrame(chunk_rows)


def _build_query_coverage_df(
    coverage_rows: List[Dict[str, Any]],
    queries: List[str],
) -> pd.DataFrame:
    """Build the Query Coverage Matrix sheet DataFrame."""
    if not coverage_rows:
        return pd.DataFrame()
    df = pd.DataFrame(coverage_rows)
    # Ensure query columns are in order
    base_cols = ["pdf"]
    query_cols = [q for q in queries if q in df.columns]
    tail_cols = ["coverage_fraction"]
    ordered = base_cols + query_cols + tail_cols
    return df[[c for c in ordered if c in df.columns]]


def build_paper_row(
    pdf_name: str,
    decision: PaperDecision,
    final_decision: str,
    evidence_gate_note: str,
    metrics: DeterministicMetrics,
    profile: PaperProfile,
    n_chunks: int,
) -> Dict[str, Any]:
    """Build one row for the Papers sheet."""
    row: Dict[str, Any] = {
        "pdf": pdf_name,
        "decision": final_decision,
        "overall_score_1_10": decision.overall_score_1_to_10,
        "composite_score_0_10": metrics.composite_score_0_to_10,
        "strong_chunks": metrics.strong_chunk_count,
        "query_coverage": metrics.query_coverage,
        "relevance_utility_gap": metrics.relevance_utility_gap,
        "paper_type": profile.paper_type,
        "domain_fit_initial": profile.domain_fit_initial,
        "main_reasoning": decision.main_reasoning,
        "strengths": " | ".join(decision.strengths),
        "weaknesses": " | ".join(decision.weaknesses),
        "best_case_for_keeping": decision.best_case_for_keeping,
        "best_case_for_discarding": decision.best_case_for_discarding,
        "agent_disagreements": " | ".join(
            d.description for d in decision.agent_disagreements
        ),
        "recommended_tags": " | ".join(decision.recommended_tags),
        "chunks_scanned": n_chunks,
        "evidence_gate_note": evidence_gate_note,
    }
    return row


def build_chunk_row(
    pdf_name: str,
    chunk: Chunk,
    a2: ChunkRelevanceQuality,
    a3: Optional[ChunkUtilityAssessment],
    chunk_weight: Optional[ChunkWeight] = None,
) -> Dict[str, Any]:
    """Build one row for the Chunks sheet.

    a3 may be None when utility simulator is skipped (A4 ablation).
    chunk_weight is populated only for the A3-w condition; other conditions
    leave the section_type/chunk_weight/weight_reasoning columns blank.
    """
    row: Dict[str, Any] = {
        "pdf": pdf_name,
        "chunk_id": chunk.chunk_id,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "domain_relevance_0_5": a2.domain_relevance_0_to_5,
        "assigned_labels": " | ".join(a2.assigned_labels),
    }
    # One column per criterion score
    for crit_name, score in a2.criteria_scores.items():
        row[f"crit_{crit_name}"] = score
    row.update({
        "figure_table_dependency": a2.figure_table_dependency,
        "standalone_comprehensibility": a2.standalone_comprehensibility,
        "contains_decision_rules": a2.contains_decision_rules,
        "contains_quantitative_data": a2.contains_quantitative_data,
        "overall_utility_0_5": a3.overall_utility_0_to_5 if a3 else None,
        "utility_type": a3.utility_type if a3 else None,
        "best_served_queries": " | ".join(a3.best_served_queries) if a3 else "",
        "evidence_snippets": " | ".join(
            f'"{s.quote}" -> {s.relevance_to_application}'
            for s in a2.evidence_snippets
        ),
        "keywords": " | ".join(a2.keywords),
        "section_type": chunk_weight.section_type if chunk_weight else "",
        "chunk_weight": chunk_weight.weight if chunk_weight else None,
        "weight_reasoning": chunk_weight.reasoning if chunk_weight else "",
    })
    return row


def build_query_coverage_row(
    pdf_name: str,
    coverage_detail: Dict[str, int],
    coverage_fraction: float,
) -> Dict[str, Any]:
    """Build one row for the Query Coverage Matrix sheet."""
    row: Dict[str, Any] = {"pdf": pdf_name}
    row.update(coverage_detail)
    row["coverage_fraction"] = coverage_fraction
    return row


def write_excel(
    paper_rows: List[Dict[str, Any]],
    chunk_rows: List[Dict[str, Any]],
    coverage_rows: List[Dict[str, Any]],
    queries: List[str],
    output_dir: Path,
    metrics_rows: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Write the multi-sheet Excel workbook (Papers, Chunks, Query Coverage, Metrics)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"filter_results_{ts}.xlsx"

    papers_df = _build_papers_df(paper_rows)
    chunks_df = _build_chunks_df(chunk_rows)
    coverage_df = _build_query_coverage_df(coverage_rows, queries)
    metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        if not papers_df.empty:
            papers_df.to_excel(writer, sheet_name="Papers", index=False)
        if not chunks_df.empty:
            chunks_df.to_excel(writer, sheet_name="Chunks", index=False)
        if not coverage_df.empty:
            coverage_df.to_excel(writer, sheet_name="Query Coverage", index=False)
        if not metrics_df.empty:
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    logger.info("Wrote Excel workbook -> %s", path)
    return path
