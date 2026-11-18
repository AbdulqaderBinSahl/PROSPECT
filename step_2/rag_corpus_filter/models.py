"""
Pydantic data models for every structured object in PROSPECT Step 2.

Matches the schemas defined in Section 5 of the Architecture Spec.
"""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 5.1  Configuration Pre-Step Output
# ---------------------------------------------------------------------------

class ValueCriterion(BaseModel):
    """A single quality criterion with rubric-style scale anchors."""
    name: str
    description: str
    scale_anchors: Dict[int, str]


class OperationalConfig(BaseModel):
    """Generated once per run by the config pre-step agent."""
    application_description: str
    domain_taxonomy: List[str]
    representative_queries: List[str]
    value_criteria: List[ValueCriterion]
    anti_patterns: List[str]
    strong_chunk_min_criteria: int
    strong_chunk_score_threshold: int
    min_strong_chunks_for_keep: int


# ---------------------------------------------------------------------------
# 5.2  Text Extraction Models
# ---------------------------------------------------------------------------

class PageText(BaseModel):
    """Raw text from a single PDF page."""
    page_num: int
    text: str


class Chunk(BaseModel):
    """A word-based text chunk with provenance."""
    chunk_id: str
    start_page: int
    end_page: int
    text: str
    word_count: int


# ---------------------------------------------------------------------------
# 5.3  Agent 1 -- Paper Profiler Output
# ---------------------------------------------------------------------------

class PaperProfile(BaseModel):
    """Factual characterization of a paper (no evaluation)."""
    paper_type: Literal[
        "experimental", "review", "simulation",
        "case_study", "methods_development",
        "industry_report", "mixed"
    ]
    methodology_summary: str
    scope: str
    data_types_present: List[Literal[
        "experimental_measurements", "process_parameters",
        "mechanical_properties", "rheological_data",
        "thermal_analysis", "spectroscopic_data",
        "microscopy_images", "simulation_results",
        "literature_compilation", "industrial_production_data",
        "none_apparent"
    ]]
    domain_fit_initial: Literal["strong", "partial", "weak"]
    profiler_notes: str


# ---------------------------------------------------------------------------
# 5.3b  ACCS — Application-Conditioned Chunk Scoring (A3-w condition)
# ---------------------------------------------------------------------------

class ChunkWeight(BaseModel):
    """Application-conditioned importance weight for a single chunk."""
    chunk_id: str
    section_type: Literal[
        "introduction", "methodology", "results", "discussion",
        "conclusion", "literature_review", "references", "other",
    ]
    weight: float = Field(..., ge=0.0, le=2.0)
    reasoning: str


class ChunkWeightsResponse(BaseModel):
    """Profiler output: per-chunk weights based on application context."""
    chunk_weights: List[ChunkWeight]


# ---------------------------------------------------------------------------
# 5.4  Agent 2 -- Relevance & Quality Assessor Output (per chunk)
# ---------------------------------------------------------------------------

class EvidenceSnippet(BaseModel):
    """A short quote from a chunk with relevance explanation."""
    quote: str
    relevance_to_application: str


class ChunkRelevanceQuality(BaseModel):
    """Agent 2 output for a single chunk."""
    domain_relevance_0_to_5: int = Field(..., ge=0, le=5)
    assigned_labels: List[str]
    anti_pattern_flags: List[str]

    criteria_scores: Dict[str, int]

    figure_table_dependency: Literal["none", "minor", "major"]
    standalone_comprehensibility: Literal[
        "fully_standalone", "mostly_standalone",
        "context_dependent", "incomprehensible_alone"
    ]

    evidence_snippets: List[EvidenceSnippet]

    contains_decision_rules: bool
    contains_mechanisms_or_causes: bool
    contains_quantitative_data: bool

    keywords: List[str]
    assessor_notes: str


# ---------------------------------------------------------------------------
# 5.5  Agent 3 -- Utility Simulator Output (per chunk)
# ---------------------------------------------------------------------------

class QueryUtilityScore(BaseModel):
    """Utility of a chunk for answering one representative query."""
    query: str
    utility_score_0_to_5: int = Field(..., ge=0, le=5)
    reasoning: str


class ChunkUtilityAssessment(BaseModel):
    """Agent 3 output for a single chunk (all queries evaluated)."""
    query_assessments: List[QueryUtilityScore]
    overall_utility_0_to_5: int = Field(..., ge=0, le=5)
    utility_type: Literal[
        "direct_answer",
        "supporting_context",
        "tangentially_related",
        "not_useful"
    ]
    best_served_queries: List[str]
    simulator_notes: str


# ---------------------------------------------------------------------------
# 5.6  Deterministic Aggregation Layer Output
# ---------------------------------------------------------------------------

class DeterministicMetrics(BaseModel):
    """Pure-computation aggregates from Agent 2 + Agent 3 outputs."""
    avg_domain_relevance: float
    avg_criteria_scores: Dict[str, float]
    strong_chunk_count: int
    anti_pattern_chunk_count: int

    avg_utility: float
    query_coverage: float
    query_coverage_detail: Dict[str, int]
    direct_answer_chunk_count: int

    relevance_utility_gap: float
    composite_score_0_to_10: float


# ---------------------------------------------------------------------------
# 5.7  Agent 4 -- Aggregator Judge Output
# ---------------------------------------------------------------------------

class AgentDisagreement(BaseModel):
    """A specific disagreement between Agent 2 and Agent 3."""
    description: str
    implication: str
    how_it_affected_decision: str


class PaperDecision(BaseModel):
    """Final Agent 4 output -- the binary keep/discard verdict."""
    decision: Literal["keep", "discard"]
    overall_score_1_to_10: int = Field(..., ge=1, le=10)

    main_reasoning: str
    strengths: List[str]
    weaknesses: List[str]

    best_case_for_keeping: str
    best_case_for_discarding: str
    why_decision_wins: str

    agent_disagreements: List[AgentDisagreement]

    recommended_tags: List[str]
    top_evidence: List[EvidenceSnippet]
    query_coverage_summary: str = ""
