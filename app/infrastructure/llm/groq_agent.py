"""Adapter: `ChatAgent` implementado via Groq + LangChain.

Traduz entre o modelo de domínio (`ChatMessage`, `AgentAnswer`) e as
estruturas do LangChain (`HumanMessage`, `AIMessage`, ...). Nenhuma regra
de negócio vive aqui — somente a mecânica de falar com a API.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from app.config.settings import AgentSettings, GroqSettings
from app.domain.exceptions import AgentError
from app.domain.models import AgentAnswer, ChatMessage, Role, Specialty
from app.infrastructure.llm.prompts import agent_system_prompt

logger = logging.getLogger(__name__)


_ROLE_TO_MESSAGE: dict[Role, type[BaseMessage]] = {
    Role.USER: HumanMessage,
    Role.ASSISTANT: AIMessage,
    Role.SYSTEM: SystemMessage,
}


def _to_langchain(message: ChatMessage) -> BaseMessage:
    """Converte `ChatMessage` do domínio para o tipo do LangChain."""
    cls = _ROLE_TO_MESSAGE[message.role]
    return cls(content=message.content)


class GroqChatAgent:
    """Implementação de `ChatAgent` usando modelos Groq via LangChain."""

    def __init__(self, groq: GroqSettings, settings: AgentSettings) -> None:
        self._settings = settings
        self._llm = ChatGroq(
            model=settings.model,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
            api_key=groq.api_key,
        )

    def answer(
        self,
        specialty: Specialty,
        history: Sequence[ChatMessage],
        question: str,
    ) -> AgentAnswer:
        system = SystemMessage(content=agent_system_prompt(specialty.name))
        past = [_to_langchain(m) for m in history]
        prompt = [system, *past, HumanMessage(content=question)]

        try:
            response = self._llm.invoke(
                prompt,
                config={"run_name": "agent-response", "tags": ["agent"]},
            )
        except Exception as exc:
            logger.exception("Falha ao chamar o agente.")
            raise AgentError(f"Falha ao obter resposta do agente: {exc}") from exc

        content = response.content
        if not isinstance(content, str):
            # ChatGroq sempre retorna str, mas o tipo é `str | list[...]`
            raise AgentError(
                f"Tipo inesperado de resposta do agente: {type(content).__name__}"
            )

        return AgentAnswer(content=content, model=self._settings.model)
