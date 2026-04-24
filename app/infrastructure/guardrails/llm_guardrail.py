"""Adapter: guardrail baseado em LLM (classificação semântica).

Segunda camada de defesa — captura conteúdo perigoso que escapa ao regex:
eufemismos, indiretas, codificações, pedidos disfarçados. Usa uma LLM
menor (modelo do agente) com saída estruturada para classificar o texto
em categorias de conteúdo perigoso.

Desenho de produção (inspirado em AWS Bedrock Guardrails + Azure Content
Safety):

1. **Prompt de classificação** com rubrica explícita e saída JSON.
2. **Confiança configurável** (`threshold`): scores acima do limiar
   bloqueiam; abaixo, permite com log de warning.
3. **Fail-open por padrão**: se a LLM do guardrail falhar, o conteúdo
   passa (prefere disponibilidade a falso positivo em massa). Em
   ambientes de alta segurança, configure `fail_closed=True`.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config.settings import GroqSettings
from app.domain.models import ContentCategory, GuardrailResult
from app.infrastructure.llm.prompts import GUARDRAIL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class _GuardrailClassification(BaseModel):
    category: Literal[
        "violence",
        "hate_speech",
        "sexual_content",
        "self_harm",
        "illegal_activity",
        "pii",
        "prompt_injection",
        "safe",
    ] = Field(
        description="Categoria do conteúdo. 'safe' se não houver violação.",
    )
    confidence: float = Field(
        description="Confiança da classificação entre 0.0 e 1.0.",
        ge=0.0,
        le=1.0,
    )
    reason: str = Field(
        description="Explicação curta da classificação em pt-BR.",
        min_length=1,
    )


class LLMGuardrail:
    """Guardrail semântico via LLM com saída estruturada.

    Usa `with_structured_output` para forçar classificação JSON — elimina
    parsing frágil de texto livre. O modelo de classificação pode ser
    diferente (e menor) que o modelo do agente para otimizar custo.
    """

    def __init__(
        self,
        *,
        groq: GroqSettings,
        model: str = "llama-3.1-8b-instant",
        threshold: float = 0.7,
        fail_closed: bool = False,
    ) -> None:
        self._threshold = threshold
        self._fail_closed = fail_closed

        llm = ChatGroq(
            model=model,
            temperature=0.0,
            max_retries=1,
            api_key=groq.api_key,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GUARDRAIL_SYSTEM_PROMPT),
                ("human", "{text}"),
            ]
        )
        self._chain = prompt | llm.with_structured_output(_GuardrailClassification)

    def check(self, text: str) -> GuardrailResult:
        try:
            result = self._chain.invoke(
                {"text": text},
                config={"run_name": "llm-guardrail", "tags": ["guardrail"]},
            )
        except Exception:
            logger.exception("Guardrail LLM falhou — fail-%s.",
                             "closed" if self._fail_closed else "open")
            if self._fail_closed:
                return GuardrailResult(
                    blocked=True,
                    category=ContentCategory.VIOLENCE,
                    reason="Guardrail indisponível e fail_closed=True.",
                )
            return GuardrailResult(blocked=False)

        if not isinstance(result, _GuardrailClassification):
            logger.warning("Guardrail retornou formato inesperado: %s",
                           type(result).__name__)
            return GuardrailResult(blocked=False)

        if result.category == "safe":
            return GuardrailResult(blocked=False)

        if result.confidence >= self._threshold:
            return GuardrailResult(
                blocked=True,
                category=ContentCategory(result.category),
                reason=result.reason,
            )

        logger.warning(
            "Guardrail: %s detectado com confiança %.2f < threshold %.2f — permitindo.",
            result.category,
            result.confidence,
            self._threshold,
        )
        return GuardrailResult(blocked=False)
