"""Tests for domain models and exceptions."""
from __future__ import annotations

import pytest

from app.domain.exceptions import (
    AgentError,
    ConfigurationError,
    DomainError,
    GuardrailBlockedError,
    JudgeError,
    ObservabilityError,
)
from app.domain.models import (
    AgentAnswer,
    ChatMessage,
    ChatTurn,
    ContentCategory,
    Evaluation,
    GuardrailResult,
    Role,
    Specialty,
)


class TestRole:
    def test_role_values(self) -> None:
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"

    def test_role_is_string_enum(self) -> None:
        assert isinstance(Role.USER, str)


class TestChatMessage:
    def test_create_user_message(self) -> None:
        msg = ChatMessage(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_message_is_frozen(self) -> None:
        msg = ChatMessage(role=Role.USER, content="Hello")
        with pytest.raises(Exception):  # dataclass frozen
            msg.content = "World"  # type: ignore


class TestSpecialty:
    def test_create_valid_specialty(self) -> None:
        specialty = Specialty(name="Python programming")
        assert specialty.name == "Python programming"

    def test_specialty_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="não pode ser vazia"):
            Specialty(name="")

    def test_specialty_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="não pode ser vazia"):
            Specialty(name="   ")

    def test_specialty_equality(self) -> None:
        s1 = Specialty(name="Python")
        s2 = Specialty(name="Python")
        assert s1 == s2


class TestAgentAnswer:
    def test_create_answer(self) -> None:
        answer = AgentAnswer(content="42 is the answer", model="llama-3.1-8b-instant")
        assert answer.content == "42 is the answer"
        assert answer.model == "llama-3.1-8b-instant"


class TestEvaluation:
    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_valid_scores(self, score: int) -> None:
        eval_ = Evaluation(score=score, justification="Good answer", judge_model="llama-3.3-70b")
        assert eval_.score == score

    @pytest.mark.parametrize("score", [0, 6, -1, 10])
    def test_invalid_scores_raise(self, score: int) -> None:
        with pytest.raises(ValueError, match="Score inválido"):
            Evaluation(score=score, justification="Good answer", judge_model="llama-3.3-70b")

    def test_empty_justification_raises(self) -> None:
        with pytest.raises(ValueError, match="justificativa não pode ser vazia"):
            Evaluation(score=5, justification="", judge_model="llama-3.3-70b")

    def test_whitespace_justification_raises(self) -> None:
        with pytest.raises(ValueError, match="justificativa não pode ser vazia"):
            Evaluation(score=5, justification="   ", judge_model="llama-3.3-70b")


class TestChatTurn:
    def test_create_turn(self) -> None:
        answer = AgentAnswer(content="Answer", model="llama-3.1-8b-instant")
        evaluation = Evaluation(score=5, justification="Correct", judge_model="llama-3.3-70b")
        specialty = Specialty(name="Math")
        turn = ChatTurn(
            question="What is 2+2?",
            answer=answer,
            evaluation=evaluation,
            specialty=specialty,
            session_id="test-session",
        )
        assert turn.question == "What is 2+2?"
        assert turn.answer == answer
        assert turn.evaluation == evaluation
        assert turn.session_id == "test-session"
        assert turn.trace_id is None
        assert turn.created_at is not None


class TestExceptions:
    def test_domain_error_inheritance(self) -> None:
        assert issubclass(DomainError, Exception)

    def test_agent_error_inheritance(self) -> None:
        assert issubclass(AgentError, DomainError)

    def test_judge_error_inheritance(self) -> None:
        assert issubclass(JudgeError, DomainError)

    def test_observability_error_inheritance(self) -> None:
        assert issubclass(ObservabilityError, DomainError)

    def test_configuration_error_inheritance(self) -> None:
        assert issubclass(ConfigurationError, DomainError)

    def test_guardrail_blocked_error_inheritance(self) -> None:
        assert issubclass(GuardrailBlockedError, DomainError)

    def test_guardrail_blocked_error_properties(self) -> None:
        err = GuardrailBlockedError(category="violence", reason="test")
        assert err.category == "violence"
        assert err.reason == "test"


class TestContentCategory:
    def test_all_categories_exist(self) -> None:
        categories = [
            ContentCategory.VIOLENCE,
            ContentCategory.HATE_SPEECH,
            ContentCategory.SEXUAL_CONTENT,
            ContentCategory.SELF_HARM,
            ContentCategory.ILLEGAL_ACTIVITY,
            ContentCategory.PII,
            ContentCategory.PROMPT_INJECTION,
        ]
        assert len(categories) == 7

    def test_category_is_string_enum(self) -> None:
        assert isinstance(ContentCategory.VIOLENCE, str)


class TestGuardrailResult:
    def test_safe_result(self) -> None:
        result = GuardrailResult(blocked=False)
        assert not result.blocked

    def test_blocked_result(self) -> None:
        result = GuardrailResult(
            blocked=True,
            category=ContentCategory.VIOLENCE,
            reason="Dangerous",
        )
        assert result.blocked
        assert result.category == ContentCategory.VIOLENCE

    def test_blocked_without_category_raises(self) -> None:
        with pytest.raises(ValueError, match="category e reason"):
            GuardrailResult(blocked=True, reason="test")

    def test_blocked_without_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="category e reason"):
            GuardrailResult(blocked=True, category=ContentCategory.VIOLENCE)

    def test_result_is_frozen(self) -> None:
        result = GuardrailResult(blocked=False)
        with pytest.raises(Exception):
            result.blocked = True  # type: ignore