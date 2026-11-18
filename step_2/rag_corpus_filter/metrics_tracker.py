"""
Metrics tracking for PROSPECT Step 2 pipeline.

Tracks per-paper timing, LLM call counts, and retry counts.
Produces per-paper dicts (embedded in decision JSONs) and a
run-level summary JSON.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from rag_corpus_filter.config import PipelineConfig


@dataclass
class PaperMetrics:
    """Accumulates metrics for a single paper."""

    pdf_name: str
    n_chunks: int = 0

    # Wall-clock timing (seconds, via perf_counter)
    paper_start: float = 0.0
    paper_end: float = 0.0
    agent1_time: float = 0.0
    agent2_total_time: float = 0.0
    agent3_total_time: float = 0.0
    agent4_time: float = 0.0

    # LLM call counts (initial + retries)
    agent1_llm_calls: int = 0
    agent2_llm_calls: int = 0
    agent3_llm_calls: int = 0
    agent4_llm_calls: int = 0

    # Retry/repair counts only
    agent1_retries: int = 0
    agent2_retries: int = 0
    agent3_retries: int = 0
    agent4_retries: int = 0

    failed: bool = False
    decision: str = ""  # "keep" / "discard" once decided

    @property
    def total_wall_time(self) -> float:
        return self.paper_end - self.paper_start

    @property
    def total_llm_calls(self) -> int:
        return (self.agent1_llm_calls + self.agent2_llm_calls
                + self.agent3_llm_calls + self.agent4_llm_calls)

    @property
    def total_retries(self) -> int:
        return (self.agent1_retries + self.agent2_retries
                + self.agent3_retries + self.agent4_retries)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a flat dict for JSON / Excel row."""
        return {
            "pdf": self.pdf_name,
            "decision": self.decision,
            "n_chunks": self.n_chunks,
            "total_wall_time_sec": round(self.total_wall_time, 2),
            "agent1_time_sec": round(self.agent1_time, 2),
            "agent2_total_time_sec": round(self.agent2_total_time, 2),
            "agent3_total_time_sec": round(self.agent3_total_time, 2),
            "agent4_time_sec": round(self.agent4_time, 2),
            "total_llm_calls": self.total_llm_calls,
            "agent1_llm_calls": self.agent1_llm_calls,
            "agent2_llm_calls": self.agent2_llm_calls,
            "agent3_llm_calls": self.agent3_llm_calls,
            "agent4_llm_calls": self.agent4_llm_calls,
            "total_retries": self.total_retries,
            "agent1_retries": self.agent1_retries,
            "agent2_retries": self.agent2_retries,
            "agent3_retries": self.agent3_retries,
            "agent4_retries": self.agent4_retries,
            "failed": self.failed,
        }


def agent_tag_from_context_label(label: str) -> str:
    """Derive which agent a call belongs to from its context_label."""
    if label.endswith("_profiler") or label.endswith("_chunk_weights"):
        return "agent1"
    if "_relevance" in label:
        return "agent2"
    if "_utility" in label:
        return "agent3"
    if label.endswith("_judge"):
        return "agent4"
    return "other"


def failed_papers(rows: List[Dict[str, Any]]) -> List[str]:
    """PDFs whose most recent attempt failed (a later successful resume clears them)."""
    latest: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        latest[r.get("pdf", "")] = r
    return [pdf for pdf, r in latest.items() if r.get("failed", False)]


def _sum_key(rows: List[Dict[str, Any]], key: str) -> float:
    """Sum a numeric key across a list of dicts, defaulting missing to 0."""
    return sum(r.get(key, 0) for r in rows)


@dataclass
class RunMetricsTracker:
    """Collects PaperMetrics for an entire pipeline run.

    On resume, call ``load_previous()`` to pull in per-paper metrics from
    earlier sessions so the final summary and Excel sheet cover all papers.
    """

    condition: str = ""
    n_pdfs: int = 0
    run_start: float = field(default_factory=time.perf_counter)
    papers: List[PaperMetrics] = field(default_factory=list)
    _historical: List[Dict[str, Any]] = field(default_factory=list)
    _prev_wall_time: float = 0.0

    def load_previous(self, output_dir: Path) -> List[Dict[str, Any]]:
        """Load per-paper metrics from run_metrics.json.

        Returns the historical per-paper dicts (also stored internally for
        merging into ``build_summary``).
        """
        path = output_dir / "run_metrics.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._historical = data.get("per_paper", [])
            self._prev_wall_time = data.get("total_wall_time_sec", 0.0)
            return self._historical
        except Exception:
            return []

    def new_paper(self, pdf_name: str) -> PaperMetrics:
        """Create and register a new PaperMetrics, stamped with start time."""
        pm = PaperMetrics(pdf_name=pdf_name, paper_start=time.perf_counter())
        self.papers.append(pm)
        return pm

    def build_summary(self, config: "PipelineConfig") -> Dict[str, Any]:
        """Build the end-of-run summary dict, merging historical data if present."""
        run_end = time.perf_counter()
        session_wall = round(run_end - self.run_start, 2)

        # Current session per-paper dicts
        current_dicts = [p.to_dict() for p in self.papers]
        # Combined: historical (previous sessions) + current
        all_dicts = self._historical + current_dicts

        successful = [r for r in all_dicts if not r.get("failed", False)]
        total_papers = len(successful)
        total_chunks = int(_sum_key(successful, "n_chunks"))
        total_llm_calls = int(_sum_key(successful, "total_llm_calls"))
        total_retries = int(_sum_key(successful, "total_retries"))
        paper_times = [r.get("total_wall_time_sec", 0) for r in successful]
        cumulative_wall = self._prev_wall_time + session_wall

        failed = failed_papers(all_dicts)
        return {
            "condition": self.condition,
            "run_timestamp": datetime.now().isoformat(),
            "total_papers": self.n_pdfs,
            "total_papers_processed": total_papers,
            "total_papers_failed": len(failed),
            "failed_papers": failed,
            "keep": sum(1 for r in successful if r.get("decision") == "keep"),
            "discard": sum(1 for r in successful if r.get("decision") == "discard"),
            "total_chunks": total_chunks,
            "total_llm_calls": total_llm_calls,
            "total_wall_time_sec": round(cumulative_wall, 2),
            "this_session_wall_time_sec": session_wall,
            "avg_time_per_paper_sec": round(
                sum(paper_times) / total_papers, 2
            ) if total_papers else 0,
            "avg_time_per_chunk_sec": round(
                sum(paper_times) / total_chunks, 2
            ) if total_chunks else 0,
            "avg_llm_calls_per_paper": round(
                total_llm_calls / total_papers, 2
            ) if total_papers else 0,
            "total_retries": total_retries,
            "hardware_model_info": {
                "model": config.llm.model,
                "base_url": config.llm.base_url,
                "temperature": config.llm.temperature,
                "max_retries": config.llm.max_retries,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "per_paper": all_dicts,
        }

    def save_summary(self, config: "PipelineConfig") -> Path:
        """Write run_metrics.json to the output dir.

        Uses a fixed filename so it can be updated incrementally after
        each paper without creating a new file every time.
        """
        summary = self.build_summary(config)
        path = config.paths.output_dir / "run_metrics.json"
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
