"""
Pydantic data models for every structured object in the pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Intent Brief
# ---------------------------------------------------------------------------

class KnowledgeAxis(BaseModel):
    """A single axis of knowledge the corpus must cover."""
    name: str = Field(..., description="Short label, e.g. 'degradation mechanisms'")
    description: str = Field(..., description="What this axis covers in one sentence")
    priority: str = Field(
        "secondary",
        description="Axis priority: 'primary', 'secondary', or 'supporting'",
    )


class IntentBrief(BaseModel):
    """Structured summary of the user's corpus-building intent."""
    domain: str = Field(..., description="Primary domain, e.g. 'polymer recycling'")
    application_context: str = Field(
        ...,
        description="How the corpus will be used, e.g. 'decision support for material selection'",
    )
    knowledge_axes: List[KnowledgeAxis] = Field(
        ..., min_length=1, description="Distinct types of knowledge the corpus must cover"
    )
    scope_boundaries: List[str] = Field(
        default_factory=list,
        description="Topics or areas explicitly excluded from the corpus (for search filtering)",
    )
    content_anti_patterns: List[str] = Field(
        default_factory=list,
        description=(
            "Full-text content patterns that indicate a paper is noise for the "
            "downstream RAG application, even if topically related. Written for "
            "chunk-level evaluation, not search exclusion."
        ),
    )
    depth_expectation: str = Field(
        ...,
        description="'survey-level breadth' or 'deep mechanistic detail' or similar",
    )
    time_window: str = Field(
        ..., description="Publication year range, e.g. '2015-2025'"
    )
    target_corpus_size: Optional[int] = Field(
        None, description="Desired number of papers (optional)"
    )


# ---------------------------------------------------------------------------
# Search Strategy
# ---------------------------------------------------------------------------

class ConceptGroup(BaseModel):
    """A group of related search terms for one concept."""
    name: str = Field(..., description="Concept label, e.g. 'polymer type'")
    knowledge_axis: str = Field(
        ..., description="Which knowledge axis this group serves"
    )
    terms: List[str] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="Primary term + synonyms/abbreviations/variants",
    )


class SearchQuery(BaseModel):
    """A single Scopus Boolean query string with metadata."""
    query_string: str = Field(..., description="Full Scopus query with field codes")
    query_type: str = Field(
        ..., description="'broad' (TITLE-ABS-KEY) or 'narrow' (TITLE)"
    )
    knowledge_axes_served: List[str] = Field(
        ..., description="Which knowledge axes this query targets"
    )
    concept_groups_used: List[str] = Field(
        ..., description="Names of concept groups combined in this query"
    )


class SearchStrategy(BaseModel):
    """Complete search strategy produced by the LLM."""
    concept_groups: List[ConceptGroup] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 concept groups",
    )
    queries: List[SearchQuery] = Field(
        ..., description="Boolean queries combining concept groups"
    )
    time_filter: str = Field(
        ..., description="Scopus time filter, e.g. 'PUBYEAR > 2015'"
    )
    subject_area_filters: List[str] = Field(
        ..., description="Scopus SUBJAREA codes, e.g. ['MATE', 'CHEM']"
    )


# ---------------------------------------------------------------------------
# Query Validation
# ---------------------------------------------------------------------------

class QueryValidationResult(BaseModel):
    """Result of a Scopus count check on a single query."""
    query_string: str
    result_count: int
    status: str = Field(
        ..., description="'ok', 'too_broad', or 'too_narrow'"
    )
    refined_query: Optional[str] = None
    refined_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Paper Metadata & Retrieval
# ---------------------------------------------------------------------------

class PaperMetadata(BaseModel):
    """Metadata for a single retrieved paper."""
    title: str
    abstract: Optional[str] = None
    authors: Optional[str] = None
    doi: Optional[str] = None
    publication_year: Optional[int] = None
    source_title: Optional[str] = None
    citation_count: Optional[int] = None
    author_keywords: Optional[str] = None
    indexed_keywords: Optional[str] = None
    document_type: Optional[str] = None
    scopus_id: Optional[str] = None
    eid: Optional[str] = None
    retrieved_by_queries: List[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Summary of what a single query retrieved."""
    query_string: str
    papers_retrieved: int
    paper_dois: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Corpus Output (top-level)
# ---------------------------------------------------------------------------

class SummaryStatistics(BaseModel):
    """Aggregate statistics about the retrieved corpus."""
    total_unique_papers: int
    papers_per_query: Dict[str, int]
    year_distribution: Dict[str, int]


class CorpusOutput(BaseModel):
    """Final output artefact of the pipeline."""
    intent_brief: IntentBrief
    search_strategy: SearchStrategy
    validation_results: List[QueryValidationResult]
    papers: List[PaperMetadata]
    query_results: List[QueryResult]
    summary: SummaryStatistics
    generated_at: str
