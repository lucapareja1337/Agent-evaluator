"""Adapter: guardrail composto que encadeia multiplos guardrails.

Padrao Chain of Responsibility: cada guardrail na cadeia avalia o texto
em sequencia. O primeiro que bloquear encerra a avaliacao imediatamente
(short-circuit). Se nenhum bloquear, o resultado e safe.

Ordem recomendada para producao:
1. RegexGuardrail - rapido, barato, deterministico
2. LLMGuardrail - semantico, mais lento, mais preciso

Isso garante que conteudo obviamente perigoso e bloqueado em microssegundos
sem chamar a LLM, enquanto conteudo ambiguo ou sofisticado e capturado
pela segunda camada.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.domain.models import GuardrailResult
from app.domain.ports import InputGuardrail, OutputGuardrail

logger = logging.getLogger(__name__)


class CompositeGuardrail:
    """Encadeia guardrails em sequencia com short-circuit no bloqueio.

    Implementa tanto InputGuardrail quanto OutputGuardrail - a
    diferenca e semantica (quem chama), nao estrutural (ambos recebem
    texto e retornam GuardrailResult).
    """

    def __init__(
        self, guardrails: Sequence[InputGuardrail | OutputGuardrail]
    ) -> None:
        self._guardrails = list(guardrails)

    def check(self, text: str) -> GuardrailResult:
        for guardrail in self._guardrails:
            result = guardrail.check(text)
            if result.blocked:
                logger.info(
                    "Guardrail bloqueou conteudo: category=%s reason=%s",
                    result.category,
                    result.reason,
                )
                return result
        return GuardrailResult(blocked=False)
