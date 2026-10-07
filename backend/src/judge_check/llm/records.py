"""Turning a CallRecord into database columns: one function for every LLM-output table, so
generation rows and judgement rows can't record calls differently (D-030, D-031)."""

from __future__ import annotations

from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from judge_check.llm.runner import CallRecord
from judge_check.models import LLMCallColumns, PromptTemplateRow
from judge_check.prompts import PromptTemplate


def ensure_prompt_row(session: Session, prompt: PromptTemplate) -> None:
    """Store the prompt text under its hash (idempotent)."""
    session.execute(
        pg_insert(PromptTemplateRow)
        .values(
            sha256=prompt.sha256,
            name=prompt.name,
            version=prompt.version,
            kind=prompt.kind,
            system=prompt.system,
            user=prompt.user,
        )
        .on_conflict_do_nothing(index_elements=["sha256"])
    )


def call_columns(session: Session, record: CallRecord) -> dict[str, Any]:
    """All LLMCallColumns values for one recorded call. Ensures the prompt row exists."""
    ensure_prompt_row(session, record.prompt)
    c = record.completion
    values = {
        "provider": c.provider,
        "model_requested": c.model_requested,
        "model_reported": c.model_reported,
        "model_version": c.model_version,
        "temperature": c.temperature,
        "seed": c.seed,
        "max_tokens": c.max_tokens,
        "prompt_sha256": record.prompt.sha256,
        "messages": [{"role": m.role, "content": m.content} for m in record.messages],
        "output_text": c.text,
        "raw_response": c.raw,
        "latency_ms": c.latency_ms,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
        "cost_usd": c.cost_usd,
    }
    assert set(LLMCallColumns.REQUIRED) <= values.keys()
    return values
