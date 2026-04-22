"""Use case: execução de um turno do chat.

O `ChatService` é o coração da aplicação — orquestra agente, juiz,
histórico persistente e observabilidade. Só conhece **portas** (Protocols
do domínio), nunca adapters concretos. Isso permite:

- Trocar o provedor de LLM (Groq → OpenAI) sem alterar nada aqui.
- Trocar a ferramenta de tracing (MLflow → Langfuse) sem alterar nada aqui.
- Trocar o backend de memória (in-memory → LangGraph/SQLite → Redis) idem.
- Testar com fakes/mocks sem tocar em rede nem disco.
"""

from __future__ import annotations

import logging
import uuid

import mlflow

from app.domain.exceptions import AgentError, JudgeError
from app.domain.models import ChatTurn, Specialty
from app.domain.ports import (
    ChatAgent,
    ConversationHistory,
    ConversationHistoryFactory,
    Judge,
    ObservabilityPort,
)

logger = logging.getLogger(__name__)


def _generate_session_id() -> str:
    """Identificador curto e legível para uma nova sessão."""
    return f"chat-{uuid.uuid4().hex[:8]}"


class ChatService:
    """Orquestra um turno completo: pergunta → resposta → avaliação → log.

    Parâmetros de construção:
        agent: Adapter do agente especialista.
        judge: Adapter do juiz.
        observability: Adapter de tracing/feedback (fail-soft).
        history_factory: Fábrica que resolve histórico por sessão.
        specialty: Especialidade declarada do agente.
        session_id: Identificador da sessão. Se ``None``, uma nova sessão é
            criada com um UUID curto. Passar um ID existente **retoma** a
            conversa, se o backend de histórico persistir esse ID.
    """

    def __init__(
        self,
        *,
        agent: ChatAgent,
        judge: Judge,
        observability: ObservabilityPort,
        history_factory: ConversationHistoryFactory,
        specialty: Specialty,
        session_id: str | None = None,
    ) -> None:
        self._agent = agent
        self._judge = judge
        self._observability = observability
        self._specialty = specialty
        self._session_id = session_id or _generate_session_id()
        self._history: ConversationHistory = history_factory.for_session(
            self._session_id
        )

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def specialty(self) -> Specialty:
        return self._specialty

    def history_length(self) -> int:
        """Número de mensagens persistidas até agora."""
        return len(self._history.load())

    # ------------------------------------------------------------------
    # Operações
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        """Limpa o histórico da sessão atual."""
        self._history.clear()

    def handle_turn(self, question: str) -> ChatTurn:
        """Executa um turno e retorna o resultado completo.

        Propaga ``AgentError`` se o agente falhar (nada a persistir).
        Propaga ``JudgeError`` se o juiz falhar — neste caso, o par
        (pergunta, resposta) **é** adicionado ao histórico para manter
        a conversa utilizável, mas o turno não é considerado completo.
        """
        if not question.strip():
            raise ValueError("A pergunta não pode ser vazia.")

        past_messages = self._history.load()

        with mlflow.start_span(
            name="chat-turn",
            attributes={
                "specialty": self._specialty.name,
                "session_id": self._session_id,
            },
        ) as span:
            span.set_inputs({"question": question})
            trace_id = span.trace_id

            try:
                answer = self._agent.answer(
                    specialty=self._specialty,
                    history=past_messages,
                    question=question,
                )
            except AgentError:
                logger.error("Agente falhou — encerrando turno.")
                raise

            try:
                evaluation = self._judge.evaluate(
                    specialty=self._specialty,
                    question=question,
                    answer=answer,
                )
            except JudgeError:
                logger.error("Juiz falhou — turno do agente preservado.")
                self._persist_turn_messages(question, answer.content)
                raise

            span.set_outputs(
                {"answer": answer.content, "score": evaluation.score}
            )

        turn = ChatTurn(
            question=question,
            answer=answer,
            evaluation=evaluation,
            specialty=self._specialty,
            session_id=self._session_id,
            trace_id=trace_id,
        )

        # Observabilidade é fail-soft — não bloqueia o chat
        self._observability.record_turn(turn)

        # Só persiste no histórico depois que agente + juiz finalizaram OK
        self._persist_turn_messages(question, answer.content)

        return turn

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _persist_turn_messages(self, question: str, answer: str) -> None:
        self._history.append_user(question)
        self._history.append_assistant(answer)
