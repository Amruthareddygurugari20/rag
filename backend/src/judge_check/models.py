"""ORM models. Each table arrives with the stage that first uses it (D-008).

Stage 1: corpus, document, question, question_evidence, chunk_set, chunk,
         embedding_run, chunk_embedding.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from judge_check.db import Base


class Corpus(Base):
    """One loaded evaluation set: its documents and its questions."""

    __tablename__ = "corpus"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (UniqueConstraint("corpus_id", "external_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    corpus_id: Mapped[int] = mapped_column(ForeignKey("corpus.id", ondelete="CASCADE"))
    external_id: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text)
    # [[start, end], ...] character spans, if the source provided sentence boundaries.
    sentence_spans: Mapped[list[list[int]] | None] = mapped_column(JSONB)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class Question(Base):
    __tablename__ = "question"
    __table_args__ = (UniqueConstraint("corpus_id", "external_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    corpus_id: Mapped[int] = mapped_column(ForeignKey("corpus.id", ondelete="CASCADE"))
    external_id: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    evidence: Mapped[list[QuestionEvidence]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )


class QuestionEvidence(Base):
    """Gold evidence as a character span of a document. Gold *chunks* are derived from this
    per chunking by interval overlap, see judge_check.retrieval.gold (D-014)."""

    __tablename__ = "question_evidence"
    __table_args__ = (CheckConstraint("char_start >= 0 AND char_end > char_start"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        ForeignKey("question.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"))
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)

    question: Mapped[Question] = relationship(back_populates="evidence")


class ChunkSet(Base):
    """One chunking of one corpus. Several can coexist, so strategies can be compared."""

    __tablename__ = "chunk_set"
    __table_args__ = (UniqueConstraint("corpus_id", "label"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    corpus_id: Mapped[int] = mapped_column(ForeignKey("corpus.id", ondelete="CASCADE"))
    label: Mapped[str] = mapped_column(Text)  # e.g. "sentence_window(sentences_per_chunk=2,...)"
    config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (
        UniqueConstraint("chunk_set_id", "document_id", "position"),
        CheckConstraint("char_start >= 0 AND char_end > char_start"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    chunk_set_id: Mapped[int] = mapped_column(ForeignKey("chunk_set.id", ondelete="CASCADE"))
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)  # 0-based order within the document
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)  # == document.text[char_start:char_end]


class EmbeddingRun(Base):
    """Which model embedded which chunk set. Vectors from different runs never mix (D-015)."""

    __tablename__ = "embedding_run"
    __table_args__ = (UniqueConstraint("chunk_set_id", "model_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_set_id: Mapped[int] = mapped_column(ForeignKey("chunk_set.id", ondelete="CASCADE"))
    model_name: Mapped[str] = mapped_column(Text)
    dimension: Mapped[int] = mapped_column(Integer)
    # True means every stored vector has unit length, so inner product == cosine (D-018).
    normalized: Mapped[bool]
    # The instruction the model expects in front of *queries* (passages get none), D-017.
    query_instruction: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embedding"

    embedding_run_id: Mapped[int] = mapped_column(
        ForeignKey("embedding_run.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunk.id", ondelete="CASCADE"), primary_key=True
    )
    # Dimension-less column so one table can hold any model; the run row records the
    # dimension and ingestion checks every vector against it (D-015).
    embedding: Mapped[Any] = mapped_column(Vector())


# --- BM25 in SQL (D-021) -------------------------------------------------------------------


class Bm25Stats(Base):
    """Corpus-level BM25 statistics for one chunk set."""

    __tablename__ = "bm25_stats"

    chunk_set_id: Mapped[int] = mapped_column(
        ForeignKey("chunk_set.id", ondelete="CASCADE"), primary_key=True
    )
    tokenizer: Mapped[str] = mapped_column(Text)
    n_docs: Mapped[int] = mapped_column(Integer)  # N: number of chunks
    avgdl: Mapped[float]  # mean chunk length in tokens
    avg_idf: Mapped[float]  # mean raw IDF over all terms, used for the epsilon floor
    epsilon: Mapped[float]


class Bm25Doc(Base):
    __tablename__ = "bm25_doc"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunk.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_set_id: Mapped[int] = mapped_column(
        ForeignKey("chunk_set.id", ondelete="CASCADE"), index=True
    )
    length: Mapped[int] = mapped_column(Integer)  # |D|: number of tokens


class Bm25Term(Base):
    __tablename__ = "bm25_term"

    chunk_set_id: Mapped[int] = mapped_column(
        ForeignKey("chunk_set.id", ondelete="CASCADE"), primary_key=True
    )
    term: Mapped[str] = mapped_column(Text, primary_key=True)
    df: Mapped[int] = mapped_column(Integer)  # n(t): chunks containing the term
    idf: Mapped[float]  # effective IDF, after the epsilon floor


class Bm25Posting(Base):
    __tablename__ = "bm25_posting"

    chunk_set_id: Mapped[int] = mapped_column(
        ForeignKey("chunk_set.id", ondelete="CASCADE"), primary_key=True
    )
    term: Mapped[str] = mapped_column(Text, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunk.id", ondelete="CASCADE"), primary_key=True
    )
    tf: Mapped[int] = mapped_column(Integer)  # f(t, D): occurrences of the term in the chunk
