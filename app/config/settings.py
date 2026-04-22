"""Configuração da aplicação.

Usa `pydantic-settings` para carregar valores do `.env` com validação
automática de tipos. Se uma variável obrigatória estiver ausente ou
inválida, a aplicação falha no startup — nunca em tempo de execução.

A separação entre `AgentSettings`, `JudgeSettings` e `MLflowSettings`
deixa explícito o escopo de cada bloco de configuração.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Configuração do agente especialista (LLM geradora)."""

    model: str = Field(
        default="llama-3.1-8b-instant",
        description="Modelo Groq usado pelo agente.",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_retries: int = Field(default=2, ge=0, le=10)

    model_config = SettingsConfigDict(env_prefix="AGENT_", extra="ignore")


class JudgeSettings(BaseSettings):
    """Configuração do juiz (LLM avaliadora)."""

    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Modelo Groq usado pelo juiz. Deve ser >= agente em qualidade.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_retries: int = Field(default=2, ge=0, le=10)

    model_config = SettingsConfigDict(env_prefix="JUDGE_", extra="ignore")


# class GroqSettings(BaseSettings):
#     """Credenciais do provedor Groq."""

#     api_key: SecretStr = Field(..., description="GROQ_API_KEY")

#     model_config = SettingsConfigDict(env_prefix="GROQ_", extra="ignore")

class GroqSettings(BaseSettings):
    """Credenciais do provedor Groq."""

    api_key: SecretStr = Field(..., description="GROQ_API_KEY")

    # ADD env_file=".env" HERE
    model_config = SettingsConfigDict(env_prefix="GROQ_", env_file=".env", extra="ignore")


class MLflowSettings(BaseSettings):
    """Configuração do backend de observabilidade."""

    tracking_uri: str = Field(
        default="http://localhost:5000",
        description="URI do MLflow Tracking Server (use 'file:./mlruns' para modo local).",
    )
    experiment_name: str = Field(default="llm-eval-system")

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")


class HistorySettings(BaseSettings):
    """Configuração do backend de memória/histórico."""

    backend: Literal["memory", "sqlite"] = Field(
        default="sqlite",
        description=(
            "Backend de persistência do histórico. 'memory' para dev/testes "
            "(não sobrevive a restart). 'sqlite' usa LangGraph + SqliteSaver."
        ),
    )
    sqlite_path: str = Field(
        default="./data/checkpoints.sqlite",
        description="Caminho do arquivo SQLite (só usado quando backend=sqlite).",
    )

    model_config = SettingsConfigDict(env_prefix="HISTORY_", extra="ignore")


class Settings(BaseSettings):
    """Agregador: expõe todas as seções como atributos.

    Ao instanciar `Settings()`, pydantic percorre cada submodelo e carrega
    as variáveis de ambiente correspondentes. Se algo falhar, `ValidationError`
    é lançado imediatamente — o que queremos.
    """

    groq: GroqSettings = Field(default_factory=GroqSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    history: HistorySettings = Field(default_factory=HistorySettings)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
