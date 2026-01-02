# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated Discord bot for the Augmentation Lab research community featuring:

- **Dual-model AI system** with cost-efficient Haiku and high-capability Sonnet models
- **Multi-provider support** (Anthropic, OpenRouter, Groq, OpenAI) with dynamic switching
- **Semantic memory system** using OpenAI embeddings for persistent conversation context
- **Autonomous strategic intelligence** with scheduled research and community analysis
- **Role-based access control** restricted to "Resident '25" Discord role
- **Modular architecture** with separate AI, storage, and management components

## Key Commands

### Development Commands
```bash
# Start the bot
./start_bot.sh

# Start manually
python auglab_bot_main.py

# Run tests
python test_auglab_bot.py
python test_functions.py

# Check bot health
python health_check.py
```

### Bot Commands (Discord)
- `!budget` - Show cost tracking and remaining budget
- `!models` - List available AI models and current configuration  
- `!switch_model <cheap|expensive> <model_name>` - Switch between AI models
- `!constitution` - View the bot's current operational constitution
- `!memory_query <query>` - Search semantic memory for past conversations
- `!memory_stats` - Display memory system statistics
- `!autonomous_think` - Trigger strategic analysis session
- `!status` - System health and capability overview

## Architecture Overview

### Core Entry Point
- **`auglab_bot_main.py`** - Main bot instance with modular component initialization
- Orchestrates all subsystems: AI models, memory, storage, function calling

### AI System (`auglab_bot/ai/`)
- **`model_manager.py`** - Multi-provider AI client management (Anthropic, OpenRouter, Groq)
- **`response_generator.py`** - Conversation handling with 24-hour context windows
- **`function_calling.py`** - Tool use framework for autonomous operations
- **`conversation_manager.py`** - Message context and conversation state management

### Storage System (`auglab_bot/storage/`)
- **`memory_system.py`** - Semantic memory using OpenAI embeddings with SQLite storage
- **`message_storage.py`** - 24-hour conversation context tracking
- **`cost_tracker.py`** - Budget management and usage analytics
- **`data_store.py`** - Channel and member data persistence

### Configuration (`auglab_bot/config/`)
- **`config.yaml`** - Comprehensive bot configuration including models, channels, budgets
- **`manager.py`** - Configuration loading with environment variable overrides
- Uses YAML anchors for model configuration reuse

### Bot Functions (`auglab_bot/bot/`)
- **`functions.py`** - Autonomous function implementations for search, analysis, member management

## Database Schema

The bot uses SQLite with these key tables:
- **`interactions`** - User conversations with embeddings
- **`memories`** - Semantic memory storage with OpenAI embeddings
- **`cost_entries`** - Budget tracking per model and user
- **`message_storage`** - 24-hour conversation context
- **`user_tags`** - Member role and identification data

## Environment Configuration

Required environment variables:
```bash
DISCORD_BOT_TOKEN=         # Discord bot token
ANTHROPIC_API_KEY=         # Anthropic Claude API key  
OPEN_ROUTER_KEY=          # OpenRouter API key
GROQ_API_KEY=             # Groq API key (optional)
OPENAI_API_KEY=           # OpenAI API key (for embeddings)
DATABASE_PATH=            # Production database path (Render deployment)
```

## Model Management

The bot supports dynamic model switching:
- **Cheap models** - Used for routine operations (Haiku, Groq models)
- **Expensive models** - Used for complex analysis (Sonnet, GPT-4, Grok)
- **Token limits** - Configurable per provider to control response length
- **Budget awareness** - Tracks costs and switches to cheaper models when budget is low

## Testing

Run comprehensive tests:
```bash
# Test bot initialization and core functions
python test_auglab_bot.py

# Test function calling system
python test_functions.py

# Manual testing with simulated messages
python simulate_message.py
python simulate_bot_mention.py
```

## Deployment

### Local Development
1. Copy `env_example.txt` to `.env` and configure API keys
2. Run `python auglab_bot_main.py` or use `./start_bot.sh`

### Production (Render)
- Uses `render.yaml` for deployment configuration
- Database migrates from repository to persistent disk on first run
- Health check endpoint available via `health_check.py`
- Automatic process management and restart capabilities

## Important Notes

- **Memory Migration**: The bot automatically migrates to OpenAI embeddings for semantic search
- **Role Restrictions**: Only users with "Resident '25" role can interact with the bot
- **Budget Limits**: Bot tracks costs and can autonomously switch to cheaper models
- **24-Hour Context**: Bot maintains conversation context across sessions using message storage
- **Autonomous Mode**: Bot can proactively analyze community needs and make strategic suggestions

## Legacy Files

The `legacy_bots/` directory contains previous bot versions for reference:
- `discord_bot.py` - Original simple bot
- `discord_bot_roleplay.py` - Roleplay-focused version
- `discord_bot_simple.py` - Minimal implementation

These are kept for historical reference but are not actively used.