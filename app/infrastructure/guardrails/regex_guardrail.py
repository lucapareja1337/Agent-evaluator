"""Adapter: guardrail baseado em padrões regex.

Primeira camada de defesa — rápido, determinístico e sem custo de LLM.
Detecta conteúdo perigoso por padrões de texto conhecidos (palavras-chave,
frases, regex compilados). Não substitui o guardrail LLM, mas o complementa:

1. RegexGuardrail bloqueia o que é *obviamente* perigoso (falso negativo
   próximo de zero, algum falso positivo aceitável).
2. LLMGuardrail captura o que escapa ao regex (semântica, eufemismos,
   indiretas).

Os padrões são compilados uma vez na construção para performance em
concorrência. Cada padrão mapeia para uma `ContentCategory` para
rastreabilidade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.models import ContentCategory, GuardrailResult

_REGEX_PATTERNS: dict[ContentCategory, tuple[str, ...]] = {
    ContentCategory.VIOLENCE: (
        r"\b(kill|murder|assassinat|tortur|massacr|bomb|shoot|stab|slaughter"
        r"|mass shooting|armed attack|suicide bomb|terrorist attack"
        r"|how to make a bomb|how to make a weapon|how to build a bomb"
        r"|como fazer uma bomba|como fazer uma arma|como matar"
        r"|assassinato|tortura|massacre|atirador|esfaquear|estupro"
        r"|viol[êe]ncia armada|ataque terrorista)"
        r"\b",
    ),
    ContentCategory.HATE_SPEECH: (
        r"\b(hate\s+speech|racial\s+slur|ethnic\s+cleansing|genocide"
        r"|white\s+supremac|nazi|neonazi|discurso de [ôo]dio"
        r"|limpeza [ée]tnica|genoc[íi]dio|supremacia racial)"
        r"\b",
    ),
    ContentCategory.SELF_HARM: (
        r"\b(suicide|self.harm|self.harm|cutting\b|overdose\b"
        r"|how to kill myself|want to die|end my life"
        r"|suic[íi]dio|automutila[çc][ãa]o|quero morrer|acabar com a vida"
        r"|como me matar|como cometer suic[íi]dio)"
        r"\b",
    ),
    ContentCategory.ILLEGAL_ACTIVITY: (
        r"\b(how to (make|cook|manufacture|synthesize)\s+"
        r"(meth|cocaine|heroin|fentanyl|drug|drugs)"
        r"|counterfeit\s+(money|currency|bill)"
        r"|money\s+laundering|tax\s+evasion\s+scheme"
        r"|como (fazer|produzir|sintetizar)\s+"
        r"(droga|coca[íi]na|hero[íi]na|metanfetamina|fentanil)"
        r"|falsifica[çc][ãa]o\s+de\s+(dinheiro|nota|moeda)"
        r"|lavagem\s+de\s+dinheiro|evadir\s+imposto)"
        r"\b",
    ),
    ContentCategory.PROMPT_INJECTION: (
        r"(?i)"
        r"(ignore\s+(all\s+)?previous\s+(instructions|prompts|rules)"
        r"|forget\s+(all\s+)?previous\s+(instructions|prompts|rules)"
        r"|you\s+are\s+now|new\s+instruction|system\s*:\s*"
        r"|override\s+(your|the)\s+(system|safety|rules)"
        r"|ignore\s+(your|the)\s+(system|safety|rules)"
        r"|disregard\s+(your|the)\s+(system|safety|rules)"
        r"|ignore\s+(all\s+)?(above|previous)\s+(instructions|directives)"
        r"|revea?l\s+(your|the|my)\s+(system|initial)\s+prompt"
        r"|what\s+is\s+your\s+system\s+prompt"
        r"|jailbreak|DAN\s+mode|developer\s+mode)"
        r"\b",
    ),
    ContentCategory.SEXUAL_CONTENT: (
        r"\b(child\s+(sexual|porn|abuse|exploitation)|csam|csae"
        r"|pedophil|incest\s+(sex|porn|content)"
        r"|abuso\s+sexual\s+infantil|pedofilia|CSAM"
        r"|explora[çc][ãa]o\s+sexual\s+infantil)"
        r"\b",
    ),
    ContentCategory.PII: (
        r"\b\d{3}[.-]?\d{3}[.-]?\d{3}[.-]?\d{2}\b"
        r"|"
        r"\b[A-Z]{2}\d{9}\b"
        r"|"
        r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
    ),
}


@dataclass(frozen=True, slots=True)
class CompiledPattern:
    pattern: re.Pattern[str]
    category: ContentCategory


def _compile_patterns(
    overrides: dict[ContentCategory, tuple[str, ...]] | None = None,
) -> list[CompiledPattern]:
    source = {**_REGEX_PATTERNS, **(overrides or {})}
    compiled: list[CompiledPattern] = []
    for category, patterns in source.items():
        for raw in patterns:
            compiled.append(
                CompiledPattern(
                    pattern=re.compile(raw, re.IGNORECASE | re.MULTILINE),
                    category=category,
                )
            )
    return compiled


class RegexGuardrail:
    """Guardrail determinístico baseado em padrões regex.

    Zero-latência, zero-custo, determinístico. Ideal como primeira camada
    antes do guardrail LLM. Compila todos os padrões na construção para
    performance em chamadas repetidas.
    """

    def __init__(
        self,
        *,
        enabled_categories: set[ContentCategory] | None = None,
        pattern_overrides: dict[ContentCategory, tuple[str, ...]] | None = None,
    ) -> None:
        all_compiled = _compile_patterns(pattern_overrides)
        if enabled_categories is not None:
            self._patterns = [
                p for p in all_compiled if p.category in enabled_categories
            ]
        else:
            self._patterns = all_compiled

    def check(self, text: str) -> GuardrailResult:
        for compiled in self._patterns:
            if compiled.pattern.search(text):
                return GuardrailResult(
                    blocked=True,
                    category=compiled.category,
                    reason=f"Padrão bloqueado pela categoria {compiled.category.value}.",
                )
        return GuardrailResult(blocked=False)
