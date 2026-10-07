"""ORM models. Each table arrives with the stage that first uses it (D-008).

Stage 1: corpus, document, question, question_evidence, chunk_set, chunk,
         embedding_run, chunk_embedding, bm25_*.
Stage 2: prompt_template, generated_answer (+ the LLMCallColumns shared with stage 4).
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
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

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


# --- LLM calls (stage 2; reused by stage 4 judgements) ----------------------------------------


class PromptTemplateRow(Base):
    """Copy of every prompt version ever used, keyed by the hash of its file (D-031).
    The repository is the source of truth; this copy keeps the database self-contained."""

    __tablename__ = "prompt_template"
    __table_args__ = (UniqueConstraint("name", "version"),)

    sha256: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(Text)
    system: Mapped[str] = mapped_column(Text)
    user: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LLMCallColumns:
    """Columns every table that stores an LLM output must have (D-030).

    The model identity and decoding parameters live on the *row*, not on a config table:
    configs change while old rows stay, and a number you can't attribute to an exact model
    version is worthless. The database enforces it (NOT NULL plus a CHECK against empty
    strings), and tests/test_llm_call_columns.py covers every table using this mixin.
    """

    REQUIRED = (
        "provider", "model_requested", "model_reported", "model_version",
        "temperature", "seed", "max_tokens", "prompt_sha256",
    )  # fmt: skip

    provider: Mapped[str] = mapped_column(Text)
    model_requested: Mapped[str] = mapped_column(Text)  # tag / deployment we asked for
    model_reported: Mapped[str] = mapped_column(Text)  # model name the provider returned
    model_version: Mapped[str] = mapped_column(Text)  # exact weights: tag@digest, dated model
    temperature: Mapped[float]
    seed: Mapped[int] = mapped_column(Integer)
    max_tokens: Mapped[int] = mapped_column(Integer)
    messages: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)  # the prompt as sent
    output_text: Mapped[str] = mapped_column(Text)
    raw_response: Mapped[dict[str, Any]] = mapped_column(JSONB)
    latency_ms: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @declared_attr
    def prompt_sha256(cls) -> Mapped[str]:  # noqa: N805
        return mapped_column(ForeignKey("prompt_template.sha256"))

    @declared_attr.directive
    def __table_args__(cls) -> tuple:  # noqa: N805
        return (
            CheckConstraint(
                "provider <> '' AND model_requested <> '' AND model_reported <> '' "
                "AND model_version <> ''",
                name=f"ck_{cls.__tablename__}_attribution",
            ),
        )


class GeneratedAnswer(LLMCallColumns, Base):
    """A grounded answer generated over retrieved chunks (stage 2).

    Its correctness is NOT known: a model wrote it. It is never a ground-truth label
    (D-000); stage 3 builds labelled variants by construction.
    """

    __tablename__ = "generated_answer"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        ForeignKey("question.id", ondelete="CASCADE"), index=True
    )
    embedding_run_id: Mapped[int] = mapped_column(ForeignKey("embedding_run.id"))
    retrieval_mode: Mapped[str] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[list[int]] = mapped_column(JSONB)  # in prompt order
    answer: Mapped[str | None] = mapped_column(Text)  # None if the output didn't parse
    cited_chunk_ids: Mapped[list[int]] = mapped_column(JSONB)
    answerable: Mapped[bool | None]
    parse_error: Mapped[str | None] = mapped_column(Text)
