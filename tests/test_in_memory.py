"""Tests for in-memory history adapter."""
from __future__ import annotations

import pytest

from app.domain.models import ChatMessage, Role
from app.infrastructure.memory.in_memory import InMemoryHistory, InMemoryHistoryFactory


class TestInMemoryHistory:
    def test_session_id(self) -> None:
        history = InMemoryHistory("test-session")
        assert history.session_id == "test-session"

    def test_load_empty_initially(self) -> None:
        history = InMemoryHistory("test")
        assert list(history.load()) == []

    def test_append_user(self) -> None:
        history = InMemoryHistory("test")
        history.append_user("Hello")
        msgs = list(history.load())
        assert len(msgs) == 1
        assert msgs[0].role == Role.USER
        assert msgs[0].content == "Hello"

    def test_append_assistant(self) -> None:
        history = InMemoryHistory("test")
        history.append_assistant("Hi there")
        msgs = list(history.load())
        assert len(msgs) == 1
        assert msgs[0].role == Role.ASSISTANT
        assert msgs[0].content == "Hi there"

    def test_clear(self) -> None:
        history = InMemoryHistory("test")
        history.append_user("Hello")
        history.append_assistant("Hi")
        history.clear()
        assert list(history.load()) == []

    def test_messages_order(self) -> None:
        history = InMemoryHistory("test")
        history.append_user("Q1")
        history.append_assistant("A1")
        history.append_user("Q2")
        history.append_assistant("A2")
        msgs = list(history.load())
        assert len(msgs) == 4
        assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]


class TestInMemoryHistoryFactory:
    def test_creates_new_history_for_new_session(self) -> None:
        factory = InMemoryHistoryFactory()
        h1 = factory.for_session("session-1")
        h2 = factory.for_session("session-2")
        assert h1 is not h2
        assert h1.session_id == "session-1"
        assert h2.session_id == "session-2"

    def test_returns_same_history_for_same_session(self) -> None:
        factory = InMemoryHistoryFactory()
        h1 = factory.for_session("session-1")
        h2 = factory.for_session("session-1")
        assert h1 is h2

    def test_sessions_are_isolated(self) -> None:
        factory = InMemoryHistoryFactory()
        h1 = factory.for_session("session-1")
        h2 = factory.for_session("session-2")
        h1.append_user("Message in session 1")
        assert list(h2.load()) == []
        assert len(list(h1.load())) == 1