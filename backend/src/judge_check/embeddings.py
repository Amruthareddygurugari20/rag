"""Text embeddings.

An embedding model maps a piece of text to a fixed-length vector (384 numbers for
bge-small-en-v1.5). It was trained so that texts which answer or paraphrase each other get
vectors pointing in similar *directions*. Retrieval is then: embed the query, find the
chunk vectors with the smallest angle to it. The cosine of that angle is the similarity.
See docs/learn/01-retrieval.md.

**Asymmetric encoding (D-017).** BGE models were trained with an instruction in front of
*queries only*. A question ("Who founded X?") and a passage that answers it ("X was founded
by Y in 1912.") are different kinds of text. The prefix tells the model "this is a search
query, put it where its answers live". Passages are embedded as they are. The prefix is
applied in exactly one place, ``embed_queries``, and can be switched off so the effect can
be measured (tests/test_embeddings_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

# From the BGE model card. Applied to queries only, never to passages.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    dimension: int
    query_instruction: str = ""


KNOWN_MODELS = {
    "BAAI/bge-small-en-v1.5": EmbeddingModelSpec(
        "BAAI/bge-small-en-v1.5", 384, BGE_QUERY_INSTRUCTION
    ),
}
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Tolerance for "unit length". float32 rounding leaves norms within ~1e-6 of 1.
NORM_TOLERANCE = 1e-3


class Embedder(Protocol):
    spec: EmbeddingModelSpec

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str], *, use_instruction: bool = True) -> np.ndarray: ...


def check_unit_norm(vectors: np.ndarray) -> np.ndarray:
    """Refuse to continue if any vector isn't unit length. We search with inner product,
    which equals cosine similarity *only* for unit vectors (D-018)."""
    norms = np.linalg.norm(vectors, axis=1)
    if vectors.size and not np.all(np.abs(norms - 1.0) < NORM_TOLERANCE):
        bad = norms[np.abs(norms - 1.0) >= NORM_TOLERANCE][:3]
        raise ValueError(f"embeddings are not unit-normalised (norms e.g. {bad})")
    return vectors


class SentenceTransformerEmbedder:
    """Local CPU embeddings via sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

        self.spec = KNOWN_MODELS.get(model_name) or EmbeddingModelSpec(model_name, 0)
        self._model = SentenceTransformer(model_name, device=device)
        # Renamed in newer sentence-transformers; support both.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        dim = get_dim()
        if self.spec.dimension not in (0, dim):
            raise ValueError(f"{model_name}: expected dimension {self.spec.dimension}, got {dim}")
        self.spec = EmbeddingModelSpec(model_name, dim, self.spec.query_instruction)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,  # unit length: cosine == inner product (D-018)
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return check_unit_norm(np.asarray(vectors, dtype=np.float32))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        # Passages: no instruction.
        return self._encode(texts)

    def embed_queries(self, texts: list[str], *, use_instruction: bool = True) -> np.ndarray:
        # Queries: the model's query instruction goes in front (the asymmetric part).
        if use_instruction and self.spec.query_instruction:
            texts = [self.spec.query_instruction + t for t in texts]
        return self._encode(texts)


@lru_cache
def get_embedder(model_name: str = DEFAULT_MODEL) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)
