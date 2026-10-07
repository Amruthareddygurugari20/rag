.PHONY: up down migrate api test lint fmt

up:        ## start Postgres+pgvector and wait until it is healthy
	docker compose up -d --wait

down:      ## stop containers (data volume is kept)
	docker compose down

migrate:   ## apply database migrations to the main database
	cd backend && uv run alembic upgrade head

api:       ## run the API with auto-reload on http://localhost:8000
	cd backend && uv run uvicorn judge_check.api.main:app --reload

test:      ## run the test suite against the test database
	cd backend && uv run pytest -v

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .

fmt:
	cd backend && uv run ruff check --fix . && uv run ruff format .
