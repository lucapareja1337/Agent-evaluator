"""Exceções específicas do domínio.

Usar exceções tipadas permite que chamadores tratem cada falha de forma
distinta, em vez de depender de strings em mensagens. Todas herdam de
`DomainError` para permitir captura genérica quando desejado.
"""

from __future__ import annotations


class DomainError(Exception):
    """Classe base para todas as exceções do domínio."""


class AgentError(DomainError):
    """Falha ao obter resposta do agente."""


class JudgeError(DomainError):
    """Falha ao obter avaliação do juiz."""


class ObservabilityError(DomainError):
    """Falha na camada de observabilidade.

    Nunca deve interromper o fluxo principal do chat — a aplicação precisa
    degradar graciosamente quando o backend de tracing está indisponível.
    """


class ConfigurationError(DomainError):
    """Configuração inválida ou ausente."""
