"""Tests for settings configuration."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config.settings import (
    AgentSettings,
    HistorySettings,
    JudgeSettings,
    MLflowSettings,
)


class TestAgentSettings:
    def test_defaults(self) -> None:
        settings = AgentSettings()
        assert settings.model == "llama-3.1-8b-instant"
        assert settings.temperature == 0.3
        assert settings.max_retries == 2

    def test_env_override(self) -> None:
        settings = AgentSettings(model="custom-model", temperature=0.5)
        assert settings.model == "custom-model"
        assert settings.temperature == 0.5


class TestJudgeSettings:
    def test_defaults(self) -> None:
        settings = JudgeSettings()
        assert settings.model == "llama-3.3-70b-versatile"
        assert settings.temperature == 0.0
        assert settings.max_retries == 2

    def test_env_override(self) -> None:
        settings = JudgeSettings(model="custom-judge", temperature=0.7)
        assert settings.model == "custom-judge"


class TestMLflowSettings:
    def test_defaults(self) -> None:
        settings = MLflowSettings()
        assert settings.tracking_uri == "http://localhost:5000"
        assert settings.experiment_name == "llm-eval-system"


class TestHistorySettings:
    def test_defaults(self) -> None:
        settings = HistorySettings()
        assert settings.backend == "sqlite"
        assert settings.sqlite_path == "./data/checkpoints.sqlite"

    def test_memory_backend(self) -> None:
        settings = HistorySettings(backend="memory")
        assert settings.backend == "memory"


class TestSettings:
    def test_agent_settings_defaults(self) -> None:
        agent = AgentSettings()
        assert agent.model == "llama-3.1-8b-instant"
        assert agent.temperature == 0.3

    def test_judge_settings_defaults(self) -> None:
        judge = JudgeSettings()
        assert judge.model == "llama-3.3-70b-versatile"
        assert judge.temperature == 0.0

    def test_mlflow_settings_defaults(self) -> None:
        mlflow = MLflowSettings()
        assert mlflow.tracking_uri == "http://localhost:5000"
        assert mlflow.experiment_name == "llm-eval-system"

    def test_history_settings_defaults(self) -> None:
        history = HistorySettings()
        assert history.backend == "sqlite"
        assert history.sqlite_path == "./data/checkpoints.sqlite"