"""
Phase 4 – Scopus Retrieval Module

Takes validated queries and retrieves full paper metadata from the Scopus
Search API.  Handles pagination, rate limiting, and cross-query deduplication.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import requests

from rag_corpus_builder import config
from rag_corpus_builder.models import (
    PaperMetadata,
    QueryResult,
    SearchStrategy,
)
from rag_corpus_builder.utils import build_full_query

logger = logging.getLogger(__name__)

# Scopus Search API fields we request
_FIELDS = ",".join([
    "dc:title",
    "dc:description",       # abstract
    "dc:creator",           # first author
    "author",               # author list
    "prism:doi",
    "prism:coverDate",
    "prism:publicationName",
    "citedby-count",
    "authkeywords",
    "dc:identifier",        # SCOPUS_ID
    "eid",
    "subtypeDescription",
    "idxterms",             # indexed keywords
])


def _parse_paper(entry: dict) -> PaperMetadata:
    """Parse a single Scopus search result entry into a PaperMetadata."""
    # Extract publication year from coverDate (format: YYYY-MM-DD)
    cover_date = entry.get("prism:coverDate", "")
    pub_year = None
    if cover_date and len(cover_date) >= 4:
        try:
            pub_year = int(cover_date[:4])
        except ValueError:
            pass

    # Citation count
    cite_str = entry.get("citedby-count")
    cite_count = None
    if cite_str is not None:
        try:
            cite_count = int(cite_str)
        except ValueError:
            pass

    # Authors: try the full author list first, fall back to dc:creator
    authors = None
    author_list = entry.get("author")
    if isinstance(author_list, list):
        names = []
        for a in author_list:
            name = a.get("authname") or a.get("ce:indexed-name", "")
            if name:
                names.append(name)
        if names:
            authors = "; ".join(names)
    if not authors:
        authors = entry.get("dc:creator")

    # Scopus ID from dc:identifier (format: "SCOPUS_ID:12345")
    raw_id = entry.get("dc:identifier", "")
    scopus_id = raw_id.replace("SCOPUS_ID:", "").strip() if raw_id else None

    # Indexed keywords (from idxterms field)
    indexed_kw = None
    raw_idx = entry.get("idxterms")
    if isinstance(raw_idx, dict):
        mainterm = raw_idx.get("mainterm", [])
        if isinstance(mainterm, list):
            terms = [t.get("$", t) if isinstance(t, dict) else str(t) for t in mainterm]
            if terms:
                indexed_kw = "; ".join(terms)
        elif isinstance(mainterm, str):
            indexed_kw = mainterm

    return PaperMetadata(
        title=entry.get("dc:title", ""),
        abstract=entry.get("dc:description"),
        authors=authors,
        doi=entry.get("prism:doi"),
        publication_year=pub_year,
        source_title=entry.get("prism:publicationName"),
        citation_count=cite_count,
        author_keywords=entry.get("authkeywords"),
        indexed_keywords=indexed_kw,
        document_type=entry.get("subtypeDescription"),
        scopus_id=scopus_id,
        eid=entry.get("eid"),
    )


def _fetch_page(
    full_query: str, start: int, count: int
) -> Tuple[List[dict], int]:
    """Fetch one page of Scopus results.

    Returns (entries, total_results).
    """
    headers = {
        "X-ELS-APIKey": config.SCOPUS_API_KEY,
        "Accept": "application/json",
    }
    if config.SCOPUS_INST_TOKEN:
        headers["X-ELS-Insttoken"] = config.SCOPUS_INST_TOKEN
    params = {
        "query": full_query,
        "count": count,
        "start": start,
        "field": _FIELDS,
    }

    resp = requests.get(
        config.SCOPUS_BASE_URL, headers=headers, params=params, timeout=60
    )
    resp.raise_for_status()
    data = resp.json()

    search_results = data.get("search-results", {})
    total = int(search_results.get("opensearch:totalResults", 0))
    entries = search_results.get("entry", [])

    # Scopus returns an error entry when no results found
    if entries and isinstance(entries, list) and "error" in entries[0]:
        return [], total

    return entries, total


SCOPUS_ABSTRACT_URL = "https://api.elsevier.com/content/abstract/scopus_id/"


def _fetch_abstract(scopus_id: str) -> str | None:
    """Fetch the abstract for a single paper via the Abstract Retrieval API."""
    headers = {
        "X-ELS-APIKey": config.SCOPUS_API_KEY,
        "Accept": "application/json",
    }
    if config.SCOPUS_INST_TOKEN:
        headers["X-ELS-Insttoken"] = config.SCOPUS_INST_TOKEN

    url = f"{SCOPUS_ABSTRACT_URL}{scopus_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Navigate the nested response structure
        core = (
            data
            .get("abstracts-retrieval-response", {})
            .get("coredata", {})
        )
        abstract = core.get("dc:description")
        if not abstract:
            # Some responses nest it differently
            abstract = core.get("dc:description", {})
            if isinstance(abstract, dict):
                abstract = abstract.get("$")
        return abstract.strip() if abstract and isinstance(abstract, str) else None
    except Exception as exc:
        logger.debug("Abstract fetch failed for %s: %s", scopus_id, exc)
        return None


def _backfill_abstracts(papers: List[PaperMetadata]) -> int:
    """Fetch abstracts from the Abstract Retrieval API for papers missing them.

    Returns the number of abstracts successfully filled.
    """
    missing = [p for p in papers if not p.abstract and p.scopus_id]
    if not missing:
        return 0

    print(f"\n  Fetching abstracts for {len(missing)} papers via Abstract Retrieval API...")

    filled = 0
    for i, paper in enumerate(missing, 1):
        if i % 50 == 0 or i == len(missing):
            print(f"    Progress: {i}/{len(missing)} ({filled} abstracts fetched)")

        abstract = _fetch_abstract(paper.scopus_id)
        if abstract:
            paper.abstract = abstract
            filled += 1

        time.sleep(config.SCOPUS_REQUEST_DELAY)

    print(f"  Abstract backfill complete: {filled}/{len(missing)} fetched.")
    return filled


def retrieve_papers(
    strategy: SearchStrategy,
) -> Tuple[List[PaperMetadata], List[QueryResult]]:
    """Retrieve paper metadata for all queries in the strategy.

    Returns (deduplicated papers, per-query results).
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: Scopus Retrieval")
    print("=" * 60)

    # DOI -> PaperMetadata for deduplication
    papers_by_doi: Dict[str, PaperMetadata] = {}
    # EID -> PaperMetadata for papers without DOI
    papers_by_eid: Dict[str, PaperMetadata] = {}
    query_results: List[QueryResult] = []

    for qi, query in enumerate(strategy.queries, 1):
        full_q = build_full_query(query, strategy)
        print(f"\n  Query {qi}/{len(strategy.queries)}: {query.query_string[:70]}...")

        # First page to get total count
        try:
            entries, total = _fetch_page(full_q, start=0, count=config.SCOPUS_RESULTS_PER_PAGE)
        except requests.RequestException as exc:
            logger.error("Scopus API error on query %d: %s", qi, exc)
            print(f"    ERROR: {exc}")
            query_results.append(QueryResult(
                query_string=query.query_string, papers_retrieved=0, paper_dois=[]
            ))
            continue

        # Cap to safety limit
        to_retrieve = min(total, config.SCOPUS_MAX_RESULTS_PER_QUERY)
        print(f"    Total available: {total} | Retrieving up to: {to_retrieve}")

        all_entries = list(entries)
        retrieved = len(all_entries)

        # Paginate
        while retrieved < to_retrieve:
            time.sleep(config.SCOPUS_REQUEST_DELAY)
            try:
                entries, _ = _fetch_page(
                    full_q,
                    start=retrieved,
                    count=config.SCOPUS_RESULTS_PER_PAGE,
                )
            except requests.RequestException as exc:
                logger.error("Pagination error at offset %d: %s", retrieved, exc)
                break

            if not entries:
                break
            all_entries.extend(entries)
            retrieved = len(all_entries)

        print(f"    Retrieved: {retrieved} entries")

        # Parse and deduplicate
        query_dois: List[str] = []
        new_count = 0
        for entry in all_entries:
            paper = _parse_paper(entry)
            doi = paper.doi
            eid = paper.eid

            # Dedup key: prefer DOI, fall back to EID
            if doi:
                if doi in papers_by_doi:
                    # Already seen — just tag the extra query
                    papers_by_doi[doi].retrieved_by_queries.append(query.query_string)
                else:
                    paper.retrieved_by_queries = [query.query_string]
                    papers_by_doi[doi] = paper
                    new_count += 1
                query_dois.append(doi)
            elif eid:
                if eid in papers_by_eid:
                    papers_by_eid[eid].retrieved_by_queries.append(query.query_string)
                else:
                    paper.retrieved_by_queries = [query.query_string]
                    papers_by_eid[eid] = paper
                    new_count += 1
                query_dois.append(eid)
            else:
                # No DOI or EID — just add (no dedup possible)
                paper.retrieved_by_queries = [query.query_string]
                papers_by_doi[f"_no_id_{id(paper)}"] = paper
                new_count += 1

        print(f"    New unique papers: {new_count}")

        query_results.append(QueryResult(
            query_string=query.query_string,
            papers_retrieved=retrieved,
            paper_dois=query_dois,
        ))

        time.sleep(config.SCOPUS_REQUEST_DELAY)

    # Merge all papers
    all_papers = list(papers_by_doi.values()) + list(papers_by_eid.values())

    print(f"\n  Total unique papers across all queries: {len(all_papers)}")

    # Backfill abstracts from Abstract Retrieval API for papers missing them
    _backfill_abstracts(all_papers)

    return all_papers, query_results
