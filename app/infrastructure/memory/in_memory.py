"""Adapter: histórico em memória.

Implementação mais simples da porta `ConversationHistory`. Útil para:
- Desenvolvimento local sem infra adicional
- Testes unitários (zero I/O)
- Cenários onde a persistência não é desejada

Uma fábrica (`InMemoryHistoryFactory`) gerencia múltiplas sessões em um
único processo, cada uma com sua própria instância isolada.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.models import ChatMessage, Role


class InMemoryHistory:
    """Histórico volátil, vive apenas enquanto o processo existir."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._messages: list[ChatMessage] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    def load(self) -> Sequence[ChatMessage]:
        return tuple(self._messages)

    def append_user(self, content: str) -> None:
        self._messages.append(ChatMessage(role=Role.USER, content=content))

    def append_assistant(self, content: str) -> None:
        self._messages.append(ChatMessage(role=Role.ASSISTANT, content=content))

    def clear(self) -> None:
        self._messages.clear()


class InMemoryHistoryFactory:
    """Fábrica: mantém uma instância por `session_id` dentro do processo."""

    def __init__(self) -> None:
        self._sessions: dict[str, InMemoryHistory] = {}

    def for_session(self, session_id: str) -> InMemoryHistory:
        history = self._sessions.get(session_id)
        if history is None:
            history = InMemoryHistory(session_id)
            self._sessions[session_id] = history
        return history
