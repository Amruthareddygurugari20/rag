"""Test doubles."""

import hashlib

import numpy as np

from judge_check.embeddings import EmbeddingModelSpec
from judge_check.retrieval.tokenize import tokenize


class HashingEmbedder:
    """Deterministic bag-of-words embedder: each token adds 1 to a hashed dimension.

    Not semantic at all, which is the point: it lets the storage and search path be tested
    without downloading a model. It honours the same contract as the real embedder:
    unit-length output and a query instruction applied to queries only.
    """

    def __init__(self, dimension: int = 64, query_instruction: str = "query: ") -> None:
        self.spec = EmbeddingModelSpec("test/hashing", dimension, query_instruction)
        self.query_calls: list[str] = []

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.spec.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.spec.dimension] += 1.0
            norm = np.linalg.norm(out[i])
            out[i] = out[i] / norm if norm else np.eye(self.spec.dimension)[0]
        return out

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: list[str], *, use_instruction: bool = True) -> np.ndarray:
        if use_instruction:
            texts = [self.spec.query_instruction + t for t in texts]
        self.query_calls.extend(texts)
        return self._encode(texts)


class UnnormalisedEmbedder(HashingEmbedder):
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return super().embed_passages(texts) * 2.0
