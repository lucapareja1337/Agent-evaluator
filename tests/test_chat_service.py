"""Tests for ChatService application layer."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.application.chat_service import ChatService, _generate_session_id
from app.domain.exceptions import AgentError, GuardrailBlockedError, JudgeError
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
from app.domain.ports import (
    ChatAgent,
    ConversationHistory,
    ConversationHistoryFactory,
    InputGuardrail,
    Judge,
    ObservabilityPort,
    OutputGuardrail,
)


class FakeAgent:
    def __init__(self, answer: AgentAnswer) -> None:
        self._answer = answer

    def answer(
        self,
        specialty: Specialty,
        history: list[ChatMessage],
        question: str,
    ) -> AgentAnswer:
        return self._answer


class FakeJudge:
    def __init__(self, evaluation: Evaluation) -> None:
        self._evaluation = evaluation

    def evaluate(self, specialty: Specialty, question: str, answer: AgentAnswer) -> Evaluation:
        return self._evaluation


class FakeObservability:
    def __init__(self) -> None:
        self.recorded_turns: list[ChatTurn] = []
        self.flushed = False

    def record_turn(self, turn: ChatTurn) -> None:
        self.recorded_turns.append(turn)

    def flush(self) -> None:
        self.flushed = True


class FakeHistory(ConversationHistory):
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._messages: list[ChatMessage] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    def load(self) -> list[ChatMessage]:
        return list(self._messages)

    def append_user(self, content: str) -> None:
        self._messages.append(ChatMessage(role=Role.USER, content=content))

    def append_assistant(self, content: str) -> None:
        self._messages.append(ChatMessage(role=Role.ASSISTANT, content=content))

    def clear(self) -> None:
        self._messages.clear()


class FakeHistoryFactory:
    def __init__(self) -> None:
        self.histories: dict[str, FakeHistory] = {}

    def for_session(self, session_id: str) -> ConversationHistory:
        if session_id not in self.histories:
            self.histories[session_id] = FakeHistory(session_id)
        return self.histories[session_id]


def _mock_mlflow_span():
    span = MagicMock()
    span.trace_id = "test-trace-123"
    span.set_inputs = MagicMock()
    span.set_outputs = MagicMock()
    return span


class TestGenerateSessionId:
    def test_format_has_prefix(self) -> None:
        sid = _generate_session_id()
        assert sid.startswith("chat-")

    def test_format_has_hex_suffix(self) -> None:
        sid = _generate_session_id()
        parts = sid.split("-")
        assert len(parts) == 2
        assert len(parts[1]) == 8


class TestChatService:
    @pytest.fixture(autouse=True)
    def mock_mlflow(self):
        with patch("app.application.chat_service.mlflow") as mock:
            mock_span = _mock_mlflow_span()
            mock.start_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock.start_span.return_value.__exit__ = MagicMock(return_value=None)
            yield mock

    def test_initialization_generates_session_id(self) -> None:
        from app.application.chat_service import ChatService

        factory = FakeHistoryFactory()
        agent = FakeAgent(AgentAnswer(content="Hi", model="test"))
        judge = FakeJudge(Evaluation(score=5, justification="OK", judge_model="test"))
        obs = FakeObservability()

        service = ChatService(
            agent=agent,
            judge=judge,
            observability=obs,
            history_factory=factory,
            specialty=Specialty(name="Test"),
        )
        assert service.session_id.startswith("chat-")

    def test_initialization_uses_provided_session_id(self) -> None:
        from app.application.chat_service import ChatService

        factory = FakeHistoryFactory()
        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Test"),
            session_id="my-session",
        )
        assert service.session_id == "my-session"

    def test_specialty_property(self) -> None:
        from app.application.chat_service import ChatService

        specialty = Specialty(name="Python")
        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=FakeHistoryFactory(),
            specialty=specialty,
        )
        assert service.specialty == specialty

    def test_handle_turn_returns_complete_turn(self) -> None:
        from app.application.chat_service import ChatService

        answer = AgentAnswer(content="42", model="test-agent")
        evaluation = Evaluation(score=5, justification="Correct", judge_model="test-judge")
        factory = FakeHistoryFactory()

        service = ChatService(
            agent=FakeAgent(answer),
            judge=FakeJudge(evaluation),
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Math"),
        )

        turn = service.handle_turn("What is 21 + 21?")

        assert turn.question == "What is 21 + 21?"
        assert turn.answer == answer
        assert turn.evaluation == evaluation
        assert turn.specialty.name == "Math"
        assert turn.session_id == service.session_id

    def test_handle_turn_persists_messages(self) -> None:
        from app.application.chat_service import ChatService

        factory = FakeHistoryFactory()
        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Test"),
        )

        service.handle_turn("Hello?")

        history = factory.histories[service.session_id]
        assert len(history._messages) == 2
        assert history._messages[0].role == Role.USER
        assert history._messages[0].content == "Hello?"
        assert history._messages[1].role == Role.ASSISTANT
        assert history._messages[1].content == "Hi"

    def test_handle_turn_empty_question_raises(self) -> None:
        from app.application.chat_service import ChatService

        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=FakeHistoryFactory(),
            specialty=Specialty(name="Test"),
        )

        with pytest.raises(ValueError, match="A pergunta não pode ser vazia"):
            service.handle_turn("")

    def test_handle_turn_whitespace_question_raises(self) -> None:
        from app.application.chat_service import ChatService

        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=FakeHistoryFactory(),
            specialty=Specialty(name="Test"),
        )

        with pytest.raises(ValueError, match="A pergunta não pode ser vazia"):
            service.handle_turn("   ")

    def test_handle_turn_agent_error_propagates(self) -> None:
        from app.application.chat_service import ChatService

        class FailingAgent:
            def answer(self, specialty, history, question):
                raise AgentError("Agent failed")

        service = ChatService(
            agent=FailingAgent(),  # type: ignore
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=FakeHistoryFactory(),
            specialty=Specialty(name="Test"),
        )

        with pytest.raises(AgentError):
            service.handle_turn("Hello?")

    def test_handle_turn_judge_error_persists_messages(self) -> None:
        from app.application.chat_service import ChatService

        class FailingJudge:
            def evaluate(self, specialty, question, answer):
                raise JudgeError("Judge failed")

        factory = FakeHistoryFactory()
        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FailingJudge(),  # type: ignore
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Test"),
        )

        with pytest.raises(JudgeError):
            service.handle_turn("Hello?")

        history = factory.histories[service.session_id]
        assert len(history._messages) == 2

    def test_history_length(self) -> None:
        from app.application.chat_service import ChatService

        factory = FakeHistoryFactory()
        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Test"),
        )

        assert service.history_length() == 0
        service.handle_turn("Hello?")
        assert service.history_length() == 2

    def test_reset_history(self) -> None:
        from app.application.chat_service import ChatService

        factory = FakeHistoryFactory()
        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Test"),
        )

        service.handle_turn("Hello?")
        assert service.history_length() == 2
        service.reset_history()
        assert service.history_length() == 0

    def test_observability_records_turn(self) -> None:
        from app.application.chat_service import ChatService

        obs = FakeObservability()
        factory = FakeHistoryFactory()

        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=obs,
            history_factory=factory,
            specialty=Specialty(name="Test"),
        )

        service.handle_turn("Hello?")

        assert len(obs.recorded_turns) == 1
        assert obs.recorded_turns[0].question == "Hello?"

    def test_input_guardrail_blocks_dangerous_question(self) -> None:
        from app.application.chat_service import ChatService

        class BlockViolence:
            def check(self, text: str) -> GuardrailResult:
                if "kill" in text.lower():
                    return GuardrailResult(
                        blocked=True,
                        category=ContentCategory.VIOLENCE,
                        reason="Violence detected",
                    )
                return GuardrailResult(blocked=False)

        factory = FakeHistoryFactory()
        agent = FakeAgent(AgentAnswer(content="Boom", model="test"))
        service = ChatService(
            agent=agent,
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Test"),
            input_guardrail=BlockViolence(),  # type: ignore
        )

        with pytest.raises(GuardrailBlockedError) as exc_info:
            service.handle_turn("How to kill someone?")

        assert exc_info.value.category == "violence"
        history = factory.histories[service.session_id]
        assert len(history._messages) == 0

    def test_input_guardrail_allows_safe_question(self) -> None:
        from app.application.chat_service import ChatService

        class BlockViolence:
            def check(self, text: str) -> GuardrailResult:
                if "kill" in text.lower():
                    return GuardrailResult(
                        blocked=True,
                        category=ContentCategory.VIOLENCE,
                        reason="Violence detected",
                    )
                return GuardrailResult(blocked=False)

        factory = FakeHistoryFactory()
        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Brasilia", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Test"),
            input_guardrail=BlockViolence(),  # type: ignore
        )

        turn = service.handle_turn("Qual a capital do Brasil?")
        assert turn.question == "Qual a capital do Brasil?"

    def test_output_guardrail_blocks_dangerous_response(self) -> None:
        from app.application.chat_service import ChatService

        class BlockViolenceOutput:
            def check(self, text: str) -> GuardrailResult:
                if "kill" in text.lower():
                    return GuardrailResult(
                        blocked=True,
                        category=ContentCategory.VIOLENCE,
                        reason="Violence in response",
                    )
                return GuardrailResult(blocked=False)

        factory = FakeHistoryFactory()
        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="You should kill them", model="test")),
            judge=FakeJudge(Evaluation(score=1, justification="Dangerous", judge_model="test")),
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Test"),
            output_guardrail=BlockViolenceOutput(),  # type: ignore
        )

        with pytest.raises(GuardrailBlockedError) as exc_info:
            service.handle_turn("What should I do?")

        assert exc_info.value.category == "violence"
        history = factory.histories[service.session_id]
        assert len(history._messages) == 0

    def test_no_guardrails_by_default(self) -> None:
        from app.application.chat_service import ChatService

        factory = FakeHistoryFactory()
        service = ChatService(
            agent=FakeAgent(AgentAnswer(content="Hi", model="test")),
            judge=FakeJudge(Evaluation(score=5, justification="OK", judge_model="test")),
            observability=FakeObservability(),
            history_factory=factory,
            specialty=Specialty(name="Test"),
        )

        turn = service.handle_turn("How to kill someone?")
        assert turn is not None