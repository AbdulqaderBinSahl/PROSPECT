"""Embedding via Ollama (or sentence-transformers fallback)."""

import logging
import time
from typing import List

from tqdm import tqdm

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Embed texts using an Ollama-served model."""

    def __init__(
        self,
        model: str = "embeddinggemma:latest",
        batch_size: int = 64,
        max_retries: int = 3,
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

        # eagerly import so we fail fast
        try:
            import ollama as _ollama

            self._client = _ollama
        except ImportError:
            raise ImportError(
                "The 'ollama' package is required. Install with: pip install ollama"
            )

        self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        """Check if the model is pulled locally; if not, pull it automatically."""
        try:
            self._client.show(self.model)
            logger.debug("Ollama model '%s' is available", self.model)
        except Exception:
            logger.info(
                "Model '%s' not found locally — pulling from Ollama (this may take a few minutes)...",
                self.model,
            )
            self._client.pull(self.model)
            logger.info("Model '%s' pulled successfully", self.model)

    # ── public API ───────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts with batching and a progress bar."""
        all_embeddings: List[List[float]] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in tqdm(
            range(0, len(texts), self.batch_size),
            total=n_batches,
            desc="Embedding chunks",
            unit="batch",
        ):
            batch = texts[i : i + self.batch_size]
            embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        response = self._client.embed(model=self.model, input=[text])
        return response["embeddings"][0]

    # ── internals ────────────────────────────────────────────────────────

    def _embed_batch_with_retry(
        self, texts: List[str]
    ) -> List[List[float]]:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.embed(
                    model=self.model, input=texts
                )
                return response["embeddings"]
            except Exception as e:
                logger.warning(
                    "Embedding batch failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                else:
                    raise RuntimeError(
                        f"Embedding failed after {self.max_retries} attempts"
                    ) from e
        return []  # unreachable, keeps type checkers happy


class SentenceTransformerEmbedder:
    """Fallback embedder using sentence-transformers (runs locally on CPU/GPU)."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for this fallback. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model
        self.batch_size = batch_size
        self._model = SentenceTransformer(model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


def get_embedder(
    model: str = "embeddinggemma:latest",
    batch_size: int = 64,
    backend: str = "ollama",
):
    """Factory — returns the appropriate embedder."""
    if backend == "ollama":
        return OllamaEmbedder(model=model, batch_size=batch_size)
    elif backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model=model, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
