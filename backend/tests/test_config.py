import pytest

from judge_check.config import Settings


def test_settings_read_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_CHECK_DATABASE_URL", "postgresql+psycopg://u:p@h:1/d")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://wrong/ignored")
    assert Settings(_env_file=None).database_url == "postgresql+psycopg://u:p@h:1/d"


def test_main_and_test_databases_differ_by_default() -> None:
    # Tests truncate the test database; it must never default to the real one.
    s = Settings(_env_file=None)
    assert s.database_url != s.test_database_url
