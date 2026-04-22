"""Adapter: `ObservabilityPort` implementado via MLflow GenAI.

Fluxo:
1. `bootstrap()` é chamado uma vez no startup: configura tracking URI,
   experimento e habilita o autolog do LangChain. A partir daí, toda
   chamada LLM feita via LangChain vira span automaticamente.
2. `record_turn()` é chamado ao final de cada turno e anexa um feedback
   (o score do juiz) ao trace — ver `mlflow.log_feedback`.

Esta implementação é deliberadamente **fail-soft**: qualquer exceção do
MLflow é capturada e logada, nunca propagada. O chat deve continuar
funcionando mesmo que o tracking server esteja fora do ar.
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from app.config.settings import MLflowSettings
from app.domain.models import ChatTurn

logger = logging.getLogger(__name__)

# Nome do feedback — mantenha estável para conseguir agregar no MLflow UI
_FEEDBACK_NAME = "llm_judge_score"


class MLflowObservability:
    """Adapter de observabilidade baseado em MLflow GenAI."""

    def __init__(self, settings: MLflowSettings) -> None:
        self._settings = settings
        self._bootstrapped = False

    def bootstrap(self) -> None:
        """Inicializa MLflow. Idempotente — pode ser chamado múltiplas vezes."""
        if self._bootstrapped:
            return
        try:
            mlflow.set_tracking_uri(self._settings.tracking_uri)
            mlflow.set_experiment(self._settings.experiment_name)
            # Autolog captura todas as chamadas LangChain como spans
            mlflow.langchain.autolog()
            self._bootstrapped = True
            logger.info(
                "MLflow inicializado: tracking_uri=%s experimento=%s",
                self._settings.tracking_uri,
                self._settings.experiment_name,
            )
        except Exception:
            logger.exception(
                "Falha ao inicializar MLflow. Tracing ficará desativado."
            )

    def record_turn(self, turn: ChatTurn) -> None:
        """Anexa o feedback do juiz ao trace correspondente.

        Não cria spans aqui — o `ChatService` já envolveu a execução do
        turno num span via `@mlflow.trace`, e o `trace_id` foi propagado.
        """
        if not turn.trace_id:
            logger.debug("Turno sem trace_id — pulando log_feedback.")
            return
        try:
            mlflow.log_feedback(
                trace_id=turn.trace_id,
                name=_FEEDBACK_NAME,
                value=turn.evaluation.score,
                rationale=turn.evaluation.justification,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE,
                    source_id=turn.evaluation.judge_model,
                ),
                metadata={
                    "specialty": turn.specialty.name,
                    "session_id": turn.session_id,
                    "agent_model": turn.answer.model,
                },
            )
        except Exception:
            # Fail-soft: logamos, mas não quebramos o chat
            logger.exception("Falha ao registrar feedback no MLflow.")

    def flush(self) -> None:
        """MLflow é síncrono por padrão; flush é no-op aqui."""
        return None
