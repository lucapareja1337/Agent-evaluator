"""Adapter: `Judge` implementado via Groq + LangChain.

Usa `with_structured_output` para forçar saída em JSON conforme schema
Pydantic — elimina a necessidade de parsing manual e erros silenciosos.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config.settings import GroqSettings, JudgeSettings
from app.domain.exceptions import JudgeError
from app.domain.models import AgentAnswer, Evaluation, Specialty
from app.infrastructure.llm.prompts import JUDGE_HUMAN_TEMPLATE, JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class _JudgeOutput(BaseModel):
    """Schema que o juiz preenche. Interno ao adapter."""

    score: Literal[1, 2, 3, 4, 5] = Field(
        description="Nota de 1 a 5."
    )
    justification: str = Field(
        description="Explicação curta (1-2 frases) da nota, em pt-BR.",
        min_length=1,
    )


class GroqJudge:
    """Implementação de `Judge` usando Groq via LangChain."""

    def __init__(self, groq: GroqSettings, settings: JudgeSettings) -> None:
        self._settings = settings
        llm = ChatGroq(
            model=settings.model,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
            api_key=groq.api_key,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", JUDGE_SYSTEM_PROMPT),
                ("human", JUDGE_HUMAN_TEMPLATE),
            ]
        )
        # `chain` garante JSON estruturado, nunca string crua
        self._chain = prompt | llm.with_structured_output(_JudgeOutput)

    def evaluate(
        self,
        specialty: Specialty,
        question: str,
        answer: AgentAnswer,
    ) -> Evaluation:
        try:
            result = self._chain.invoke(
                {
                    "specialty": specialty.name,
                    "question": question,
                    "answer": answer.content,
                },
                config={"run_name": "llm-judge", "tags": ["judge"]},
            )
        except Exception as exc:
            logger.exception("Falha ao chamar o juiz.")
            raise JudgeError(f"Falha ao avaliar resposta: {exc}") from exc

        if not isinstance(result, _JudgeOutput):
            # with_structured_output pode retornar dict se o schema falhar
            raise JudgeError(
                f"Juiz retornou formato inesperado: {type(result).__name__}"
            )

        return Evaluation(
            score=result.score,
            justification=result.justification,
            judge_model=self._settings.model,
        )
