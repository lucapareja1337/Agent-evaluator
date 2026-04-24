"""Composition Root.

Único lugar do sistema que *constrói* os adapters concretos e os injeta
nas camadas superiores. Esta é a essência do padrão Dependency Injection:
toda a resolução de dependências acontece aqui, de cima para baixo, e só
aqui. Mudar um provedor (Groq → OpenAI, MLflow → Langfuse, in-memory →
LangGraph) significa mudar apenas este arquivo.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from app.application.chat_service import ChatService
from app.config.settings import HistorySettings, Settings
from app.domain.exceptions import ConfigurationError
from app.domain.models import Specialty
from app.domain.ports import ConversationHistoryFactory, InputGuardrail, OutputGuardrail
from app.infrastructure.guardrails.composite_guardrail import CompositeGuardrail
from app.infrastructure.guardrails.llm_guardrail import LLMGuardrail
from app.infrastructure.guardrails.regex_guardrail import RegexGuardrail
from app.infrastructure.llm.groq_agent import GroqChatAgent
from app.infrastructure.llm.groq_judge import GroqJudge
from app.infrastructure.memory.in_memory import InMemoryHistoryFactory
from app.infrastructure.memory.langgraph_history import LangGraphHistoryFactory
from app.infrastructure.observability.mlflow_tracer import MLflowObservability
from app.presentation.cli import ChatCLI

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    """Configuração mínima mas estruturada de logging."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _load_settings() -> Settings:
    """Carrega e valida settings. Falha cedo com mensagem útil."""
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigurationError(
            f"Configuração inválida. Verifique seu .env.\n{exc}"
        ) from exc


def _build_history_factory(
    settings: HistorySettings,
) -> ConversationHistoryFactory:
    """Instancia a fábrica de histórico conforme o backend escolhido.

    Retorna o Protocol `ConversationHistoryFactory` — chamadores não
    precisam saber qual implementação foi selecionada.
    """
    if settings.backend == "memory":
        logger.info("Histórico: backend em memória (volátil).")
        return InMemoryHistoryFactory()

    # backend == "sqlite"
    db_path = Path(settings.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    factory = LangGraphHistoryFactory(str(db_path))
    factory.open()
    logger.info("Histórico: LangGraph + SQLite em %s", db_path)
    return factory


def _ask_specialty() -> Specialty:
    """Pergunta ao usuário a especialidade do agente."""
    raw = input("\n🎓 Qual a especialidade do agente? ").strip()
    try:
        return Specialty(name=raw)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def _ask_session_to_resume(
    factory: ConversationHistoryFactory,
) -> str | None:
    """Se houver sessões anteriores, oferece retomar uma delas.

    Retorna o `session_id` escolhido ou `None` para nova sessão.
    Só é chamado quando o backend persistente expõe `list_sessions`.
    """
    list_fn = getattr(factory, "list_sessions", None)
    if list_fn is None:
        return None
    sessions = list_fn()
    if not sessions:
        return None

    print("\n📚 Sessões anteriores encontradas:")
    for idx, sid in enumerate(sessions, start=1):
        print(f" [{idx}] {sid}")
    print(" [n] Nova sessão (padrão)")

    choice = input("Escolha uma opção: ").strip().lower()
    if not choice or choice == "n":
        return None
    try:
        idx = int(choice)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]
    except ValueError:
        pass
    # qualquer entrada inválida cai em nova sessão
    print("Opção inválida — iniciando nova sessão.")
    return None


def _build_guardrails(
    settings: Settings,
) -> tuple[InputGuardrail | None, OutputGuardrail | None]:
    """Constrói a cadeia de guardrails conforme configuração.

    A ordem importa: RegexGuardrail (rápido, barato) roda antes do
    LLMGuardrail (lento, semântico) para economizar chamadas LLM
    em conteúdo obviamente perigoso.

    Retorna (input_guardrail, output_guardrail) — ambos podem ser None
    se os guardrails estiverem desabilitados.
    """
    gs = settings.guardrail
    if not gs.enabled:
        logger.info("Guardrails: desabilitados.")
        return None, None

    layers: list[InputGuardrail | OutputGuardrail] = []

    if gs.regex_enabled:
        layers.append(RegexGuardrail())
        logger.info("Guardrails: camada regex ativa.")

    if gs.llm_enabled:
        layers.append(
            LLMGuardrail(
                groq=settings.groq,
                model=gs.llm_model,
                threshold=gs.llm_threshold,
                fail_closed=gs.fail_closed,
            )
        )
        logger.info(
            "Guardrails: camada LLM ativa (model=%s, threshold=%.2f).",
            gs.llm_model,
            gs.llm_threshold,
        )

    if not layers:
        logger.info("Guardrails: nenhuma camada ativa.")
        return None, None

    composite = CompositeGuardrail(layers)
    return composite, composite


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main() -> int:
    """Entry point. Retorna código de saída no estilo Unix."""
    try:
        settings = _load_settings()
    except ConfigurationError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    _configure_logging(settings.log_level)

    # Observabilidade primeiro — autolog do LangChain precisa estar ativo
    # antes dos adapters de LLM serem instanciados
    observability = MLflowObservability(settings.mlflow)
    observability.bootstrap()

    # Backend de memória: pode precisar de cleanup (SQLite conn), então
    # guardamos em variável para `finally` garantir o close.
    history_factory = _build_history_factory(settings.history)

    try:
        try:
            specialty = _ask_specialty()
            session_id = _ask_session_to_resume(history_factory)
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Encerrando.")
            return 0
        except ConfigurationError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 2

        agent = GroqChatAgent(groq=settings.groq, settings=settings.agent)
        judge = GroqJudge(groq=settings.groq, settings=settings.judge)

        input_guardrail, output_guardrail = _build_guardrails(settings)

        service = ChatService(
            agent=agent,
            judge=judge,
            observability=observability,
            history_factory=history_factory,
            specialty=specialty,
            session_id=session_id,
            input_guardrail=input_guardrail,
            output_guardrail=output_guardrail,
        )

        cli = ChatCLI(service=service, settings=settings)
        cli.run()
    finally:
        # Ordem de teardown: flush obs → fecha SQLite (se for o caso)
        observability.flush()
        close_fn = getattr(history_factory, "close", None)
        if callable(close_fn):
            close_fn()

    return 0


if __name__ == "__main__":
    sys.exit(main())
