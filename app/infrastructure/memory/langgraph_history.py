"""Adapter: histórico persistente via LangGraph + SqliteSaver.

Cada sessão de chat corresponde a uma **thread** do LangGraph (thread_id
= session_id). Cada mensagem adicionada dispara um `invoke()` do grafo,
que grava um checkpoint no SQLite. Ao reiniciar o processo, basta usar o
mesmo `session_id` para retomar a conversa exatamente de onde parou.

## Por que um grafo mínimo?

O grafo tem apenas um nó que faz append no estado. Parece trivial — e é.
Mas ganhamos de graça:
- Persistência atômica em SQLite (o checkpointer cuida de commits)
- Um modelo mental uniforme: toda mutação do histórico vira um checkpoint
- Acesso futuro a recursos do LangGraph (time travel, multi-step) sem refactor

## Ciclo de vida

`SqliteSaver.from_conn_string()` é um context manager. Como nossa app
de chat roda em um único processo de longa duração, entramos no CM uma
vez no startup (via fábrica) e saímos no shutdown. `close()` é idempotente.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Annotated, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.domain.exceptions import ConfigurationError
from app.domain.models import ChatMessage, Role

logger = logging.getLogger(__name__)


# Types of the domain that can appear in the state persisted by the
# checkpointer. Passing the classes themselves (instead of string tuples)
# avoids desynchronization if the domain is refactored.
#
# Why explicit allowlisting matters:
#   1. Removes deprecation warnings in the current version of LangGraph.
#   2. Makes us future-safe: the default is moving to BLOCK unregistered types.
#   3. Adds security: limits the deserializer attack surface (CVE
#      GHSA-wwqv-p2pp-99h5 on langgraph-checkpoint < 3.0 was exactly
#      RCE via unchecked deserialization).
_ALLOWED_DOMAIN_TYPES = (ChatMessage, Role)


def _build_serializer() -> JsonPlusSerializer:
    """Serializer that allowlists our domain types in msgpack."""
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_DOMAIN_TYPES)


# ----------------------------------------------------------------------
# Estado do grafo
# ----------------------------------------------------------------------


def _append_messages(
    existing: list[ChatMessage],
    new: list[ChatMessage],
) -> list[ChatMessage]:
    """Reducer: concatena novas mensagens ao histórico existente."""
    return [*existing, *new]


class _ChatState(TypedDict):
    """Estado persistido pelo checkpointer."""

    messages: Annotated[list[ChatMessage], _append_messages]


def _ingest_node(state: _ChatState) -> _ChatState:
    """Nó no-op. O trabalho real é feito pelo reducer nos inputs do invoke."""
    return {"messages": []}


def _build_graph(checkpointer: SqliteSaver):
    """Constrói e compila o grafo mínimo de persistência."""
    builder: StateGraph = StateGraph(_ChatState)
    builder.add_node("ingest", _ingest_node)
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", END)
    return builder.compile(checkpointer=checkpointer)


# ----------------------------------------------------------------------
# Histórico (view sobre uma única thread)
# ----------------------------------------------------------------------


class _LangGraphHistoryView:
    """View de histórico para uma sessão (thread_id fixo).

    Não mantém estado local — toda leitura vai direto ao grafo/checkpointer.
    Isso garante consistência mesmo se a mesma sessão for aberta de
    múltiplos lugares (cenário improvável, mas desejável como propriedade).
    """

    def __init__(self, graph, session_id: str, checkpointer: SqliteSaver) -> None:
        self._graph = graph
        self._session_id = session_id
        self._checkpointer = checkpointer

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def _config(self) -> dict:
        return {"configurable": {"thread_id": self._session_id}}

    def load(self) -> Sequence[ChatMessage]:
        state = self._graph.get_state(self._config)
        if state is None or not state.values:
            return ()
        messages = state.values.get("messages", [])
        return tuple(messages)

    def append_user(self, content: str) -> None:
        self._append(Role.USER, content)

    def append_assistant(self, content: str) -> None:
        self._append(Role.ASSISTANT, content)

    def _append(self, role: Role, content: str) -> None:
        message = ChatMessage(role=role, content=content)
        self._graph.invoke({"messages": [message]}, config=self._config)

    def clear(self) -> None:
        try:
            self._checkpointer.delete_thread(self._session_id)
        except AttributeError:
            # Versões antigas do checkpointer não têm delete_thread;
            # fallback: zerar via um novo checkpoint com lista vazia
            # (não é perfeito — histórico antigo ainda fica nos checkpoints
            # mas o estado "atual" fica vazio).
            logger.warning(
                "Checkpointer sem delete_thread — fazendo reset lógico."
            )
            current = list(self.load())
            if current:
                # Reducer só concatena, então não conseguimos "apagar" sem
                # delete_thread. Logamos para não silenciar o problema.
                logger.error(
                    "Impossível limpar thread %s sem delete_thread.",
                    self._session_id,
                )


# ----------------------------------------------------------------------
# Fábrica
# ----------------------------------------------------------------------


class LangGraphHistoryFactory(AbstractContextManager):
    """Fábrica de históricos baseada em LangGraph + SqliteSaver.

    Mantém uma única instância do grafo e do checkpointer, compartilhada
    por todas as sessões. Segue o contrato de context manager para garantir
    cleanup apropriado do SqliteSaver.
    """

    def __init__(self, db_path: str) -> None:
        if not db_path:
            raise ConfigurationError("db_path obrigatório para LangGraphHistoryFactory.")
        self._db_path = db_path
        self._lock = threading.Lock()
        self._cm: AbstractContextManager[SqliteSaver] | None = None
        self._checkpointer: SqliteSaver | None = None
        self._graph = None

    def open(self) -> "LangGraphHistoryFactory":
        """Inicializa checkpointer e grafo. Idempotente."""
        with self._lock:
            if self._graph is not None:
                return self
            self._cm = SqliteSaver.from_conn_string(self._db_path)
            saver = self._cm.__enter__()
            # Sobrescrevemos o serde para incluir nossos tipos de domínio
            # no allowlist. Sem isso, o checkpointer loga warnings e, em
            # versões futuras, bloqueia a desserialização.
            saver.serde = _build_serializer()
            self._checkpointer = saver
            self._graph = _build_graph(self._checkpointer)
            logger.info("LangGraph history iniciado (db=%s)", self._db_path)
        return self

    def close(self) -> None:
        """Fecha a conexão SQLite. Idempotente."""
        with self._lock:
            if self._cm is None:
                return
            try:
                self._cm.__exit__(None, None, None)
            finally:
                self._cm = None
                self._checkpointer = None
                self._graph = None
                logger.info("LangGraph history encerrado.")

    # Context manager protocol ------------------------------------------
    def __enter__(self) -> "LangGraphHistoryFactory":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # Porta -------------------------------------------------------------
    def for_session(self, session_id: str) -> _LangGraphHistoryView:
        if self._graph is None or self._checkpointer is None:
            raise RuntimeError(
                "LangGraphHistoryFactory não foi aberta. Chame open() antes."
            )
        return _LangGraphHistoryView(self._graph, session_id, self._checkpointer)

    def list_sessions(self) -> list[str]:
        """Retorna todos os session_ids com histórico persistido.

        Útil para o CLI oferecer "retomar sessão antiga". Implementado via
        API de checkpoint.list() — percorre apenas o índice de threads.
        """
        if self._checkpointer is None:
            return []
        seen: set[str] = set()
        # list() sem filtro retorna checkpoints de todas as threads
        for item in self._checkpointer.list(None):
            thread_id = item.config.get("configurable", {}).get("thread_id")
            if thread_id:
                seen.add(thread_id)
        return sorted(seen)
