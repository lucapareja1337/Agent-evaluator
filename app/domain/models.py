"""Entidades do domínio.

Objetos puros que representam os conceitos de negócio. Não dependem de
nenhum framework externo — apenas da biblioteca padrão. São imutáveis
(`frozen=True`) para evitar efeitos colaterais inesperados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal


class Role(str, Enum):
    """Papel de uma mensagem em uma conversa."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Uma única mensagem em uma conversa."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Specialty:
    """A especialidade declarada do agente.

    Value object: duas `Specialty` com o mesmo nome são iguais.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("A especialidade não pode ser vazia.")


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """Resposta produzida pelo agente especialista."""

    content: str
    model: str


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Avaliação produzida pela LLM juiz.

    `score` é restrito ao intervalo 1..5 para manter a escala consistente
    em todo o sistema. A validação ocorre no construtor — falhar cedo.
    """

    score: Literal[1, 2, 3, 4, 5]
    justification: str
    judge_model: str

    def __post_init__(self) -> None:
        if self.score not in (1, 2, 3, 4, 5):
            raise ValueError(f"Score inválido: {self.score}. Deve estar em 1..5.")
        if not self.justification.strip():
            raise ValueError("A justificativa não pode ser vazia.")


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """Um turno completo do chat: pergunta, resposta e avaliação.

    Unidade agregadora do domínio. É o que persistimos (conceitualmente)
    e o que a camada de observabilidade consome.
    """

    question: str
    answer: AgentAnswer
    evaluation: Evaluation
    specialty: Specialty
    session_id: str
    trace_id: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ContentCategory(str, Enum):
    """Categorias de conteúdo perigoso que os guardrails devem bloquear.

    Seguem as taxonomias padrão da indústria (AWS Bedrock Guardrails,
    Azure Content Safety, Google Perspective API). Cada categoria mapeia
    para um conjunto de padrões e regras de detecção.
    """

    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL_CONTENT = "sexual_content"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    PII = "pii"
    PROMPT_INJECTION = "prompt_injection"


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Resultado da avaliação de um guardrail.

    Imutável por design: a decisão de bloqueio é atômica e não deve ser
    alterada após a avaliação. Quando `blocked=True`, `category` e
    `reason` são obrigatórios para rastreabilidade e auditoria.
    """

    blocked: bool
    category: ContentCategory | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.blocked and (self.category is None or self.reason is None):
            raise ValueError(
                "GuardrailResult bloqueado deve informar category e reason."
            )
