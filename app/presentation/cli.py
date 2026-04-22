"""Interface de linha de comando.

Responsabilidade única: I/O com o usuário. Chama o `ChatService` e formata
a saída. Não conhece Groq, MLflow, nem prompts — tudo vem injetado.
"""

from __future__ import annotations

import logging

from app.application.chat_service import ChatService
from app.config.settings import Settings
from app.domain.exceptions import AgentError, JudgeError
from app.domain.models import ChatTurn

logger = logging.getLogger(__name__)

_EXIT_COMMANDS = frozenset({"sair", "exit", "quit"})
_CLEAR_COMMANDS = frozenset({"limpar", "clear"})


class ChatCLI:
    """Loop interativo do chat no terminal."""

    def __init__(self, service: ChatService, settings: Settings) -> None:
        self._service = service
        self._settings = settings

    def run(self) -> None:
        self._print_header()

        while True:
            try:
                question = input("💬 Você: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Encerrando.")
                break

            if not question:
                continue

            command = question.lower()
            if command in _EXIT_COMMANDS:
                print("👋 Até mais!")
                break
            if command in _CLEAR_COMMANDS:
                self._service.reset_history()
                print("🧹 Histórico limpo.\n")
                continue

            self._process_turn(question)

        self._print_footer()

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _process_turn(self, question: str) -> None:
        try:
            turn = self._service.handle_turn(question)
        except AgentError as exc:
            print(f"⚠️  O agente falhou: {exc}\n")
            return
        except JudgeError as exc:
            print(f"⚠️  Avaliação indisponível neste turno: {exc}\n")
            return

        self._print_turn(turn)

    @staticmethod
    def _print_turn(turn: ChatTurn) -> None:
        print(f"\n🤖 Agente: {turn.answer.content}\n")
        print(
            f"⚖️  Juiz: {turn.evaluation.score}/5 — "
            f"{turn.evaluation.justification}\n"
        )

    def _print_header(self) -> None:
        print("=" * 60)
        print("🤖 Sistema de Chat com Avaliação Automática (LLM-as-Judge)")
        print("=" * 60)
        existing = self._service.history_length()
        resume_info = (
            f" (retomando, {existing} msgs anteriores)" if existing else ""
        )
        print(
            f"\n✅ Agente pronto (especialidade: "
            f"{self._service.specialty.name}).\n"
            f"   Agente:   {self._settings.agent.model}\n"
            f"   Juiz:     {self._settings.judge.model}\n"
            f"   Memória:  {self._settings.history.backend}\n"
            f"   Sessão:   {self._service.session_id}{resume_info}\n"
            f"   MLflow:   {self._settings.mlflow.tracking_uri}\n"
            f"   Comandos: 'sair' / 'limpar'\n"
        )

    def _print_footer(self) -> None:
        print(
            f"\n📊 Veja os resultados em {self._settings.mlflow.tracking_uri} "
            f"(experimento: {self._settings.mlflow.experiment_name}, "
            f"sessão: {self._service.session_id})"
        )
