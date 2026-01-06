# Discord Symbiont Framework

This repo is a framework for creating a Discord symbiont: a strategic agent that lives in a Discord server, interprets context, and takes goal-directed actions.

The prompt and metrics pertain specifically to Dax (see the intro [here](https://c4554ndr4.github.io/Discord_Symbiont/)), but the symbiont framework is open-source and available on [GitHub](https://github.com/c4554ndr4/Discord_Symbiont). The framework is modeled after the Cyborgism Act 1 [project](https://manifund.org/projects/act-i-exploring-emergent-behavior-from-multi-ai-multi-human-interaction). I've implemented a budget-aware routing layer; integrations for web search and image generation; commands to reason, query memory, retrieve user interaction history, and define a constitution separate from the system prompt; and a memory layer storing interaction summaries and embeddings for recall.

## What is a Discord agent?

A Discord agent is a bot that can read messages, reason about context, and take actions inside a server. Dax is an agent that combines:

- **Event-driven message handling** (responds to mentions, DMs, or allowed channels).
- **LLM routing** (cheap model for assessment, expensive model for complex tasks).
- **Tools** (search, messaging, image generation, and code execution).
- **Memory** (stores structured interaction data for recall).

## Message flow

1. **Discord event** arrives (mention, DM, or allowed channel).
2. **Assessment pass** chooses a cheap vs expensive model and sets a token budget.
3. **System + context** assembled (system prompt, recent messages, and relevant memories).
4. **Response pass** generates content or tool calls.
5. **Actions** executed (message send, search, image generation).
6. **Memory + cost** recorded (token usage, summaries, and embeddings).

## System prompts

There are two main prompt surfaces:

- **Runtime prompt** in `config.yaml` under `system_prompt.main_prompt`.
- **Project prompt** in the repo (the “main prompt” published on GitHub Pages in `docs/index.md`).

These define Dax’s identity, priorities, safety constraints, and communication style. The runtime prompt can be edited for different deployments without touching code.

## Model routing (cheap vs expensive)

The bot uses a two-stage routing strategy to control cost and latency while preserving quality:

- **Assessment model**: A lower-cost model evaluates intent, complexity, and required tools. It assigns a priority score and estimated token budget.
- **Response model**: A higher-capability model is used only when the request exceeds thresholds (complex reasoning, multi-step tool use, or high-impact replies).

Routing decisions consider:

- **Input length and context size** (system + history + memory tokens).
- **Estimated output budget** (max tokens per provider in `models.token_limits`).
- **Cost tracker limits** (per-day and per-user budgets in `cost_management`).

This keeps routine interactions fast and cheap while reserving expensive model calls for high-value responses.

## Configuration guide

The bot uses `config.yaml` for runtime behavior. A safe template lives in `config.example.yaml`. This is a really convenient paradigm for vibe-coding that I highly recommend!

- `api_keys`: Environment variable mapping for providers (Discord, Anthropic, OpenAI, OpenRouter, Groq).
- `bot_name`: Display name used in logs and interfaces.
- `channels`:
  - `auto_response_allowed`: Channels Dax can respond in without mention.
  - `mention_response_allowed`: Channels Dax can respond in when mentioned.
  - `user_messaging_allowed`: Channels Dax can send messages to proactively.
  - `excluded_channels`: Channels Dax will ignore.
- `cost_management`: Per-day and per-user limits plus alert threshold.
- `debug`: Enable extra logging and diagnostic behavior.
- `features`: Toggles for DMs, reactions, conversation memory, and slash commands.
- `functions`: Tool access policy and max tool calls per message.
- `bot`: Command prefix, target guild ID, and residency role configuration.
- `autonomous`: Scheduling and tool limits for autonomous investigations.
- `guild`: Member change logging and verification rules.
- `logging`: Log levels, file size limits, and log rotation.
- `memory`: Storage limits, cleanup intervals, and on-disk persistence.
- `models`: Provider configs, model lists, and token limits per provider.
- `response`: Max response length, retries, and typing delay.
- `system_prompt`: Runtime system prompt content.

## Features

- Dual-model responses with budget-aware routing.
- Semantic memory and searchable channel history.
- Role-aware member tools and targeted messaging.
- Built-in cost tracking and safety limits.
- Optional tool integrations (web search, image generation, code execution).

## Boot Dax locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets and settings

```bash
cp env_example.txt .env
cp config.example.yaml config.yaml
```

Edit `.env` and `config.yaml` with your tokens, guild ID, and channel lists.

### 3. Start the bot

```bash
python discord_bot_auglab.py
```

You can also use:

```bash
./start_bot.sh
```