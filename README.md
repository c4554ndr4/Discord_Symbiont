# Dax — Augmentation Lab Symbiont

Dax is the Augmentation Lab symbiont: a strategic Discord bot built to help the lab grow stronger by improving social momentum, project completion, and member satisfaction.

For the public-facing description (with links, images, and the main prompt), see `docs/index.md` and enable GitHub Pages from the `docs/` folder.

## Features

- Dual-model responses with budget-aware routing.
- Semantic memory and searchable channel history.
- Role-aware member tools and targeted messaging.
- Built-in cost tracking and safety limits.
- Optional tool integrations (web search, image generation, code execution).

## How it works

- **Assessment pass** decides whether a request is cheap or expensive.
- **Response pass** uses the selected model to answer or take actions.
- **Memory layer** stores interaction summaries and embeddings for recall.
- **Cost tracker** logs token usage and enforces budgets.

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
