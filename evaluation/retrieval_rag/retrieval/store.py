"""Chroma vector store helpers — uses HttpClient to a running Chroma server."""

import logging
import sys
from typing import Any, Dict, List, Set

import chromadb

logger = logging.getLogger(__name__)


# ── client / server ──────────────────────────────────────────────────────────


def get_client(host: str = "localhost", port: int = 8000) -> chromadb.HttpClient:
    """Return an HttpClient connected to a running Chroma server."""
    client = chromadb.HttpClient(host=host, port=port)
    return client


def check_server(client: chromadb.HttpClient) -> bool:
    """Verify that the Chroma server is reachable. Exits with clear error if not."""
    try:
        client.heartbeat()
        logger.info("Chroma server is reachable")
        return True
    except Exception as e:
        logger.error(
            "Cannot reach Chroma server: %s\n"
            "Start it in another terminal, for example:\n"
            "  chroma run --path runs/chroma --port 8000",
            e,
        )
        sys.exit(1)


# ── collection management ───────────────────────────────────────────────────


def get_or_create_collection(
    client: chromadb.HttpClient,
    name: str,
) -> chromadb.Collection:
    """Get or create a Chroma collection with cosine distance."""
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "Collection '%s' ready (%d existing documents)", name, collection.count()
    )
    return collection


def list_collections(client: chromadb.HttpClient) -> List[str]:
    """Return names of all collections on the server."""
    return [c.name for c in client.list_collections()]


# ── resume support ───────────────────────────────────────────────────────────


def get_indexed_paper_ids(collection: chromadb.Collection) -> Set[str]:
    """Return the set of paper_ids already present in the collection."""
    count = collection.count()
    if count == 0:
        return set()

    # Fetch metadatas in pages to avoid huge single response
    page_size = 5000
    paper_ids: Set[str] = set()
    offset = 0

    while offset < count:
        batch = collection.get(
            include=["metadatas"],
            limit=page_size,
            offset=offset,
        )
        for meta in batch["metadatas"]:
            if meta and "paper_id" in meta:
                paper_ids.add(meta["paper_id"])
        offset += page_size

    logger.info("Found %d already-indexed papers in collection", len(paper_ids))
    return paper_ids


# ── add / query ──────────────────────────────────────────────────────────────


def add_chunks(
    collection: chromadb.Collection,
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    """Add a batch of chunks to the collection."""
    # Chroma has a per-call limit (~5461 for large embeddings).
    # Batch ourselves to be safe.
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )


def query_collection(
    collection: chromadb.Collection,
    query_embedding: List[float],
    top_k: int = 10,
) -> Dict[str, Any]:
    """Query the collection and return formatted results.

    Returns a dict with keys: chunks (list of dicts with text, score, metadata).
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Chroma cosine distance = 1 - cosine_similarity
        similarity = 1.0 - distance
        chunks.append(
            {
                "chunk_text": meta.get("original_text", doc),
                "similarity_score": round(similarity, 4),
                "paper_id": meta.get("paper_id", ""),
                "paper_title": meta.get("title", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "source_section": meta.get("source_section", "unknown"),
                "corpus_condition": meta.get("corpus_condition", ""),
            }
        )

    return {"chunks": chunks}
