# Dax — Augmentation Lab Symbiont

Dax is the Augmentation Lab symbiont: a strategic Discord bot built to help the lab grow stronger by improving social momentum, project completion, and member satisfaction.

For the public-facing description (with links, images, and the main prompt), see `docs/index.md` and enable GitHub Pages from the `docs/` folder.

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

## Configuration guide

The bot uses `config.yaml` for runtime behavior. A safe template lives in `config.example.yaml`.

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

## Notes for public release

- Do not commit `.env`, `config.yaml`, or any `*.db` files.
- The local database stores conversation history and memory embeddings.
