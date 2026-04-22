"""Portas (interfaces) do domínio.

Seguem o padrão *Ports and Adapters* (Hexagonal Architecture): o domínio
declara o **contrato** que precisa; a infraestrutura fornece **adapters**
concretos. Isso inverte a dependência — o domínio nunca importa da
infraestrutura.

Usamos `typing.Protocol` em vez de ABCs porque:
1. Permite *structural subtyping* (duck typing estático), sem exigir herança.
2. Adapters podem ser criados por terceiros sem importar este módulo.
3. Integra-se de forma natural com mypy/pyright.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.domain.models import (
    AgentAnswer,
    ChatMessage,
    ChatTurn,
    Evaluation,
    Specialty,
)


@runtime_checkable
class ChatAgent(Protocol):
    """Agente especialista que responde a perguntas.

    Implementações típicas: adapters que encapsulam uma LLM (Groq, OpenAI...).
    """

    def answer(
        self,
        specialty: Specialty,
        history: Sequence[ChatMessage],
        question: str,
    ) -> AgentAnswer:
        """Produz uma resposta para a pergunta no contexto dado."""
        ...


@runtime_checkable
class Judge(Protocol):
    """Avaliador automático (LLM-as-a-Judge) de respostas."""

    def evaluate(
        self,
        specialty: Specialty,
        question: str,
        answer: AgentAnswer,
    ) -> Evaluation:
        """Avalia a resposta do agente segundo uma rubrica interna."""
        ...


@runtime_checkable
class ConversationHistory(Protocol):
    """Histórico de mensagens de uma sessão de chat.

    Abstrai a estratégia de persistência (memória, SQLite via LangGraph
    checkpointer, Redis, etc.). Cada instância opera sobre uma **única**
    sessão identificada no momento da criação.

    A abstração deliberadamente **não expõe** operações de baixo nível de
    checkpointing — a camada de aplicação não precisa saber que existem
    checkpoints. Isso mantém o domínio desacoplado do LangGraph.
    """

    @property
    def session_id(self) -> str:
        """Identificador da sessão que este histórico representa."""
        ...

    def load(self) -> Sequence[ChatMessage]:
        """Retorna as mensagens da sessão na ordem cronológica."""
        ...

    def append_user(self, content: str) -> None:
        """Adiciona uma mensagem do usuário ao final do histórico."""
        ...

    def append_assistant(self, content: str) -> None:
        """Adiciona uma resposta do assistente ao final do histórico."""
        ...

    def clear(self) -> None:
        """Remove todas as mensagens desta sessão."""
        ...


@runtime_checkable
class ConversationHistoryFactory(Protocol):
    """Fábrica de `ConversationHistory` por sessão.

    Necessária porque o `ChatService` não deve conhecer como se constrói
    um histórico — só pede um para a sessão atual. Permite abrir múltiplas
    sessões (threads no vocabulário do LangGraph) sem acoplamento.
    """

    def for_session(self, session_id: str) -> ConversationHistory:
        """Retorna o histórico de uma sessão (criando se não existir)."""
        ...


@runtime_checkable
class ObservabilityPort(Protocol):
    """Camada de observabilidade (tracing + feedback).

    Desacopla o domínio de ferramentas específicas (MLflow, Langfuse,
    LangSmith, OTel puro). Implementações devem ser *fail-soft*: nunca
    propagar exceções para o chamador — logar e seguir.
    """

    def record_turn(self, turn: ChatTurn) -> None:
        """Registra um turno completo (pergunta + resposta + avaliação)."""
        ...

    def flush(self) -> None:
        """Garante que todos os eventos em buffer foram persistidos."""
        ...
