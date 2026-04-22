# Sistema de Avaliação de LLM (LLM-as-a-Judge)

Chat de terminal onde um **agente especialista** responde ao usuário e cada
resposta é avaliada automaticamente por uma **LLM juiz**. A conversa é
**persistida em disco** — você pode fechar o programa e retomar depois.

- **Provedor LLM:** Groq (tier gratuito)
- **Orquestração:** LangChain
- **Memória:** LangGraph + SqliteSaver (checkpointer)
- **Observabilidade:** MLflow GenAI

## Arquitetura

Clean Architecture com *Ports & Adapters*. O domínio é puro; a
infraestrutura é plugável.

```
app/
├── domain/                  ← Regras e contratos (zero deps externas)
│   ├── models.py                Entidades imutáveis
│   ├── ports.py                 Protocols: ChatAgent, Judge,
│   │                            ConversationHistory, Observability
│   └── exceptions.py            Hierarquia de erros tipados
│
├── application/             ← Casos de uso (depende só de domain)
│   └── chat_service.py          Orquestra um turno do chat
│
├── infrastructure/          ← Adapters (dependem de libs externas)
│   ├── llm/
│   │   ├── groq_agent.py        ChatAgent via Groq
│   │   ├── groq_judge.py        Judge via Groq (structured output)
│   │   └── prompts.py           System prompts isolados
│   ├── memory/
│   │   ├── in_memory.py         Histórico volátil (dev/testes)
│   │   └── langgraph_history.py Histórico persistente (LangGraph+SQLite)
│   └── observability/
│       └── mlflow_tracer.py     MLflow + log_feedback (LLM_JUDGE)
│
├── presentation/
│   └── cli.py                   Loop interativo no terminal
│
├── config/
│   └── settings.py              Pydantic Settings (valida .env no startup)
│
└── main.py                  ← Composition Root (única DI do sistema)
```

### Fluxo de um turno

```
 usuário digita
       │
       ▼
┌─────────────────────────────────────────────┐
│  ChatService.handle_turn(question)          │
│                                             │
│   1. Carrega histórico da sessão atual      │
│   2. Abre span MLflow (chat-turn)           │
│   3. ChatAgent.answer(...)                  │
│   4. Judge.evaluate(...)                    │
│   5. Fecha span → obtém trace_id            │
│   6. ObservabilityPort.record_turn(turn)    │
│   7. history.append_user + append_assistant │
│      └── SqliteSaver grava checkpoint       │
└─────────────────────────────────────────────┘
```

### Por que dois modelos?

Padrão comum em produção: usar um modelo **pequeno e barato** (aqui
`llama-3.1-8b-instant`) para responder ao usuário, e um modelo **maior**
(aqui `llama-3.3-70b-versatile`) apenas como juiz. Como o juiz roda uma
vez por turno, o custo total segue baixo e a qualidade da avaliação
sobe significativamente.

### Memória via LangGraph

Cada `session_id` do nosso domínio mapeia para um `thread_id` do LangGraph.
Um `StateGraph` mínimo de um nó faz *append* na lista de mensagens; o
`SqliteSaver` grava um checkpoint a cada invocação. Isso dá de graça:

- **Persistência atômica** em SQLite
- **Retomada de sessão** após restart
- **Time travel** (acessível via API do LangGraph, se você quiser estender)

## Pré-requisitos (Fedora KDE 42)

```bash
sudo dnf install -y python3-pip python3-virtualenv git
python3 --version  # precisa ser >= 3.10
```

## Instalação

### 1. Ambiente virtual

```bash
cd llm-eval-system
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configurar `.env`

```bash
cp .env.example .env
```

Edite `.env` e preencha ao menos o `GROQ_API_KEY`. Obtenha em
<https://console.groq.com/keys> (gratuito, sem cartão).

### 3. Iniciar o MLflow

Em outro terminal (ou em background):

```bash
source .venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000
```

A UI fica em <http://127.0.0.1:5000>.

Se não quiser um servidor separado, basta setar
`MLFLOW_TRACKING_URI=file:./mlruns` no `.env` — os traces vão para o
disco local e podem ser visualizados depois com
`mlflow ui --backend-store-uri file:./mlruns`.

## Uso

```bash
source .venv/bin/activate
python -m app.main
```

Na primeira execução:
1. O programa pergunta qual a **especialidade** do agente.
2. Se já houver sessões anteriores no SQLite, oferece retomar alguma.
3. Inicia o loop de chat.

### Comandos do chat

- `sair` / `exit` — encerra
- `limpar` — apaga o histórico da sessão atual (via `delete_thread` no
  checkpointer)

### Exemplo de sessão

```
🎓 Qual a especialidade do agente? história do Brasil

💬 Você: quem proclamou a república?

🤖 Agente: A República foi proclamada por Marechal Deodoro da Fonseca
   em 15 de novembro de 1889...

⚖️  Juiz: 5/5 — Resposta factualmente correta, dentro do escopo.

💬 Você: qual a melhor receita de lasanha?

🤖 Agente: Essa pergunta foge da minha especialidade em história do Brasil...

⚖️  Juiz: 5/5 — O agente corretamente se manteve no escopo.

💬 Você: sair
```

Da próxima vez que você rodar:

```
📚 Sessões anteriores encontradas:
  [1] chat-a4f12b09
  [n] Nova sessão (padrão)
Escolha uma opção: 1
```

O chat retoma exatamente de onde parou.

## Onde vejo os resultados?

Acesse a UI do MLflow em <http://127.0.0.1:5000>:

- **Traces**: cada turno como um trace `chat-turn` com spans filhos
  (agente + juiz) capturados via `mlflow.langchain.autolog()`
- **Feedback**: score 1–5 e justificativa do juiz anexados como
  `Assessment` do tipo `LLM_JUDGE`, filtráveis por sessão/especialidade
- **Filtros úteis**: `feedback.llm_judge_score < 3` para revisar falhas

## Trocando componentes

A arquitetura foi desenhada para permitir substituições cirúrgicas:

| Quero trocar...               | Mexo só em...                         |
|-------------------------------|---------------------------------------|
| Groq → OpenAI                 | `infrastructure/llm/` + `main.py`     |
| MLflow → Langfuse/LangSmith   | `infrastructure/observability/` + `main.py` |
| SQLite → Postgres (LangGraph) | `infrastructure/memory/` + `main.py`  |
| CLI → API REST                | `presentation/` + `main.py`           |

Nenhuma dessas mudanças toca em `domain/` ou `application/`.

## Qualidade de código

```bash
pip install ruff mypy
ruff check app/
mypy app/
```

Configuração em `pyproject.toml`.

## Limitações conhecidas

- O reducer do `StateGraph` só concatena mensagens; `limpar` depende do
  `delete_thread` do checkpointer (disponível no `langgraph-checkpoint-sqlite`
  recente).
- `SqliteSaver` é síncrono — OK para CLI. Para uma API com carga, considere
  `AsyncSqliteSaver` ou `PostgresSaver`.
- Observabilidade é fail-soft: se o MLflow estiver fora, o chat continua
  rodando, mas você perde o registro daquele turno.
