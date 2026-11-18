"""
Phase 5 – Output Handler Module

Assembles the final CorpusOutput, computes summary statistics, and writes
the structured JSON file, the Markdown run log, the intent brief, and the
PDF checklist that tells the user which files Step 2 expects.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rag_corpus_builder import config
from rag_corpus_builder.models import (
    CorpusOutput,
    IntentBrief,
    PaperMetadata,
    QueryResult,
    QueryValidationResult,
    SearchStrategy,
    SummaryStatistics,
)

logger = logging.getLogger(__name__)


def _compute_summary(
    papers: List[PaperMetadata],
    query_results: List[QueryResult],
) -> SummaryStatistics:
    """Compute aggregate statistics over the retrieved corpus."""
    # Papers per query
    papers_per_query: Dict[str, int] = {}
    for qr in query_results:
        papers_per_query[qr.query_string[:80]] = qr.papers_retrieved

    # Year distribution
    year_counter: Counter[str] = Counter()
    for p in papers:
        year_key = str(p.publication_year) if p.publication_year else "unknown"
        year_counter[year_key] += 1

    # Sort year distribution
    year_distribution = dict(sorted(year_counter.items()))

    return SummaryStatistics(
        total_unique_papers=len(papers),
        papers_per_query=papers_per_query,
        year_distribution=year_distribution,
    )


def save_intent_brief(intent_brief: IntentBrief, out_dir: Path) -> Path:
    """Write the confirmed intent brief to ``out_dir/intent_brief.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "intent_brief.json"
    path.write_text(
        json.dumps(intent_brief.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Intent brief saved to %s", path)
    return path


def save_output(
    intent_brief: IntentBrief,
    strategy: SearchStrategy,
    validation_results: List[QueryValidationResult],
    papers: List[PaperMetadata],
    query_results: List[QueryResult],
    out_dir: Path,
) -> str:
    """Assemble and save the full corpus output to ``out_dir/corpus.json``.

    Returns the path to the saved JSON file.
    """
    print("\n" + "=" * 60)
    print("  PHASE 5: Saving Output")
    print("=" * 60)

    summary = _compute_summary(papers, query_results)

    corpus = CorpusOutput(
        intent_brief=intent_brief,
        search_strategy=strategy,
        validation_results=validation_results,
        papers=papers,
        query_results=query_results,
        summary=summary,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = str(out_dir / "corpus.json")

    # Write JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(corpus.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info("Output saved to %s", filepath)

    # Print summary
    print(f"\n  Output file: {filepath}")
    print("\n  --- Summary ---")
    print(f"  Total unique papers: {summary.total_unique_papers}")
    print(f"  Queries executed:    {len(query_results)}")

    if summary.year_distribution:
        print("\n  Year distribution:")
        for year, count in summary.year_distribution.items():
            bar = "#" * min(count // 2, 40)
            print(f"    {year}: {count:>5}  {bar}")

    print("\n  Papers per query:")
    for q_label, count in summary.papers_per_query.items():
        print(f"    {q_label}: {count}")

    return filepath


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------

def save_run_log(
    intent_brief: IntentBrief,
    transcript: List[Dict[str, str]],
    model: str,
    llm_base_url: str,
    out_dir: Path,
    strategy: Optional[SearchStrategy] = None,
    validation_results: Optional[List[QueryValidationResult]] = None,
    papers: Optional[List[PaperMetadata]] = None,
    query_results: Optional[List[QueryResult]] = None,
    corpus_filepath: Optional[str] = None,
) -> str:
    """Save a human-readable Markdown run log to ``out_dir/run_log.md``.

    Captures the full elicitation conversation, the confirmed intent brief,
    search strategy, validation results, and retrieval summary — everything
    needed to reproduce or cite the run in a paper. When Scopus retrieval
    was skipped, only the conversation and intent brief are written.

    Returns the path to the saved Markdown file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = str(out_dir / "run_log.md")

    lines: List[str] = []

    # Header
    lines.append("# PROSPECT Step 1 — Run Log\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**LLM model:** {model}")
    lines.append(f"**LLM endpoint:** {llm_base_url}")
    if corpus_filepath:
        lines.append(f"**Corpus output:** `{os.path.basename(corpus_filepath)}`")
    else:
        lines.append("**Scopus retrieval:** skipped (intent brief only)")
    lines.append("")

    # ---- Elicitation transcript ----
    lines.append("---\n")
    lines.append("## 1. Elicitation Conversation\n")

    for msg in transcript:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"**User:**\n> {content}\n")
        elif role == "assistant":
            # Truncate very long JSON-only responses for readability
            if content.strip().startswith("{") and len(content) > 500:
                lines.append("**Assistant:** *(generated Intent Brief JSON — see below)*\n")
            else:
                lines.append(f"**Assistant:**\n{content}\n")

    # ---- Confirmed Intent Brief ----
    lines.append("---\n")
    lines.append("## 2. Confirmed Intent Brief\n")
    lines.append("```json")
    lines.append(json.dumps(intent_brief.model_dump(), indent=2, ensure_ascii=False))
    lines.append("```\n")

    if strategy is not None:
        # ---- Search Strategy ----
        lines.append("---\n")
        lines.append("## 3. Search Strategy\n")
        lines.append(f"**Concept groups ({len(strategy.concept_groups)}):**\n")
        for cg in strategy.concept_groups:
            lines.append(f"- **{cg.name}** (axis: {cg.knowledge_axis}): {', '.join(cg.terms)}")
        lines.append("")

        lines.append(f"**Queries ({len(strategy.queries)}):**\n")
        for i, q in enumerate(strategy.queries, 1):
            lines.append(f"{i}. `[{q.query_type.upper()}]` `{q.query_string}`")
            lines.append(f"   - Axes: {', '.join(q.knowledge_axes_served)}")
        lines.append("")

        lines.append(f"**Time filter:** `{strategy.time_filter}`")
        lines.append(f"**Subject areas:** {', '.join(strategy.subject_area_filters)}")
        lines.append("")

    if validation_results is not None:
        # ---- Validation Results ----
        lines.append("---\n")
        lines.append("## 4. Query Validation Results\n")
        lines.append("| # | Status | Count | Query (truncated) |")
        lines.append("|---|--------|------:|-------------------|")
        for i, vr in enumerate(validation_results, 1):
            lines.append(
                f"| {i} | {vr.status} | {vr.result_count:,} | "
                f"`{vr.query_string[:70]}{'...' if len(vr.query_string) > 70 else ''}` |"
            )
        lines.append("")

    if papers is not None and query_results is not None:
        # ---- Retrieval Summary ----
        lines.append("---\n")
        lines.append("## 5. Retrieval Summary\n")
        lines.append(f"- **Total unique papers:** {len(papers)}")
        lines.append(f"- **Queries executed:** {len(query_results)}")
        lines.append("")

        lines.append("**Papers per query:**\n")
        for qr in query_results:
            lines.append(f"- `{qr.query_string[:70]}`: {qr.papers_retrieved} papers")
        lines.append("")

        # Year distribution
        year_counter: Counter[str] = Counter()
        for p in papers:
            year_key = str(p.publication_year) if p.publication_year else "unknown"
            year_counter[year_key] += 1
        if year_counter:
            lines.append("**Year distribution:**\n")
            lines.append("| Year | Count |")
            lines.append("|------|------:|")
            for year, count in sorted(year_counter.items()):
                lines.append(f"| {year} | {count} |")
            lines.append("")

    # ---- Pipeline Config ----
    lines.append("---\n")
    lines.append("## 6. Pipeline Configuration\n")
    lines.append(f"- LLM temperature: {config.LLM_TEMPERATURE}")
    lines.append(f"- LLM max tokens: {config.LLM_MAX_TOKENS}")
    lines.append(f"- Max clarification rounds: {config.MAX_CLARIFICATION_ROUNDS}")
    lines.append(f"- Max validation iterations: {config.MAX_VALIDATION_ITERATIONS}")
    lines.append(f"- Scopus results per page: {config.SCOPUS_RESULTS_PER_PAGE}")
    lines.append(f"- Scopus max results per query: {config.SCOPUS_MAX_RESULTS_PER_QUERY}")
    lines.append("")

    # Write
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Run log saved to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# PDF checklist (hand-off to Step 2)
# ---------------------------------------------------------------------------

def expected_pdf_filename(index: int, doi: Optional[str], scopus_id: Optional[str]) -> str:
    """Suggested PDF filename that Step 2 can match back to its metadata.

    Step 2 matches a PDF to its Step 1 record when the filename contains the
    paper's DOI (or Scopus ID), ignoring punctuation. The ``Paper<N>_``
    prefix keeps files in corpus order.
    """
    if doi:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", doi).strip("_")
        return f"Paper{index}_{slug}.pdf"
    if scopus_id:
        return f"Paper{index}_scopus_{scopus_id}.pdf"
    return f"Paper{index}_NO_IDENTIFIER.pdf"


def write_pdf_checklist(corpus_path: Path, out_dir: Path) -> Path:
    """Write ``pdf_checklist.csv`` listing every paper in a corpus JSON.

    Each row gives the filename Step 2 expects, so the user can save the
    PDFs they download under those names in the run's ``papers/`` folder.
    """
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    papers = corpus.get("papers", [])
    path = out_dir / "pdf_checklist.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "n", "expected_filename", "doi", "doi_link", "title",
            "publication_year", "source_title", "document_type", "note",
        ])
        for i, p in enumerate(papers, 1):
            doi = p.get("doi")
            fname = expected_pdf_filename(i, doi, p.get("scopus_id"))
            note = "" if (doi or p.get("scopus_id")) else "no DOI or Scopus ID: Step 2 cannot match this PDF to its abstract"
            writer.writerow([
                i, fname, doi or "", f"https://doi.org/{doi}" if doi else "",
                p.get("title", ""), p.get("publication_year") or "",
                p.get("source_title") or "", p.get("document_type") or "", note,
            ])
    logger.info("PDF checklist (%d papers) saved to %s", len(papers), path)
    return path
