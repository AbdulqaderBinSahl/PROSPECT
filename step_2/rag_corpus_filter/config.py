"""
Configuration for PROSPECT Step 2 pipeline.

All settings -- LLM, paths, chunking defaults -- in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Path defaults (relative to step_2/)
# ---------------------------------------------------------------------------

_STEP2_ROOT = Path(__file__).resolve().parent.parent  # step_2/

DEFAULT_PAPERS_DIR = _STEP2_ROOT / "papers"
DEFAULT_OUTPUT_DIR = _STEP2_ROOT / "output"
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class LLMSettings:
    """OpenAI-compatible endpoint settings (local server or cloud API).

    ``run_step2.py`` overwrites model / base_url / api_key with the values
    resolved from the CLI and .env; the defaults below are the settings
    used in the PROSPECT paper (gpt-oss-120b served by LM Studio).
    """
    model: str = "openai/gpt-oss-120b"
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "not-needed"
    temperature: float = 0.1
    max_tokens: int = 4096  # prevent thinking-mode models from exhausting budget on reasoning
    max_retries: int = 10
    retry_delay_sec: float = 0.5


@dataclass
class AgentModelMapping:
    """Per-agent model assignment for multi-model setups.

    When set, each agent uses its own ChatOpenAI client pointed at the
    specified model.  All models must be loaded on the same LM Studio
    server (same base_url).  If a field is None, falls back to
    LLMSettings.model (single-model mode).
    """
    agent1_profiler: str | None = None
    agent2_relevance: str | None = None
    agent3_utility: str | None = None
    agent4_aggregator: str | None = None
    config_prestep: str | None = None


@dataclass
class ChunkingSettings:
    """Word-based chunking parameters."""
    chunk_words: int = 1400
    overlap_words: int = 200
    min_chunk_words: int = 300


@dataclass
class PathSettings:
    """All filesystem paths used by the pipeline."""
    papers_dir: Path = field(default_factory=lambda: DEFAULT_PAPERS_DIR)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    prompts_dir: Path = field(default_factory=lambda: DEFAULT_PROMPTS_DIR)

    @property
    def routed_pdfs_dir(self) -> Path:
        return self.output_dir / "routed_pdfs"

    @property
    def keep_dir(self) -> Path:
        return self.routed_pdfs_dir / "keep"

    @property
    def discard_dir(self) -> Path:
        return self.routed_pdfs_dir / "discard"

    @property
    def paper_decisions_dir(self) -> Path:
        return self.output_dir / "paper_decisions"

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [
            self.output_dir,
            self.keep_dir,
            self.discard_dir,
            self.paper_decisions_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    llm: LLMSettings = field(default_factory=LLMSettings)
    agent_models: AgentModelMapping = field(default_factory=AgentModelMapping)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    profiler_n_pages: int = 2  # first/last N pages for Agent 1
