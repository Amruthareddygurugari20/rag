"""FastAPI application. Run with: uv run uvicorn judge_check.api.main:app --reload"""

from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from judge_check import __version__
from judge_check.db import get_session, pgvector_version

app = FastAPI(title="judge-check", version=__version__)


@app.get("/health")
def health(session: Annotated[Session, Depends(get_session)]) -> JSONResponse:
    """Liveness plus the two things every later stage depends on: the DB and pgvector."""
    try:
        version = pgvector_version(session)
    except OperationalError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "database": "unreachable", "detail": str(exc.orig)},
        )
    if version is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "database": "ok",
                "pgvector": "missing: run `uv run alembic upgrade head`",
            },
        )
    return JSONResponse(content={"status": "ok", "database": "ok", "pgvector": version})
