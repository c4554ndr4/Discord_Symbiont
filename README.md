# Dax — Augmentation Lab Symbiont

Dax is the Augmentation Lab symbiont: a strategic Discord bot built to help the lab grow stronger by improving social momentum, project completion, and member satisfaction.

For the public-facing description (with links, images, and the main prompt), see `docs/index.md` and enable GitHub Pages from the `docs/` folder.

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
- Rotate any API keys that were ever committed to git history.
