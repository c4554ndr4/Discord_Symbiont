# Augmentation Lab Discord Bot 🧪

A sophisticated Discord bot designed specifically for the Augmentation Lab community, featuring intelligent cost tracking, dual-model AI responses, comprehensive data storage, and advanced tool integration.

## 🌟 Features

### Core Capabilities
- **Intelligent AI Responses**: Uses Haiku for initial assessment, then Sonnet 4 for complex tasks
- **Budget Management**: Robust $100 compute budget tracking with real-time monitoring
- **Complete Data Storage**: Stores all channel messages for future reference and search
- **Member Management**: Special handling for '25 residency group members
- **MCP Tool Integration**: Web search, image generation, code execution, and file operations

### AI Models & Cost Optimization
- **Claude 3 Haiku**: Fast, cheap initial processing and importance assessment
- **Claude 3.5 Sonnet**: Advanced reasoning for complex tasks and important responses
- **GPT-4o**: Image generation and analysis capabilities
- **Smart Model Selection**: Automatically chooses the most appropriate model

### Advanced Tools
- **Web Search**: Real-time information retrieval via DuckDuckGo
- **Image Generation**: DALL-E 3 integration for creating visuals
- **Code Assistance**: Safe code execution and debugging support
- **File Operations**: Secure file management capabilities

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the bot files
# Install dependencies
pip install -r requirements.txt

# Run the setup script
python setup_auglab_bot.py
```

### 2. Configuration

1. **Edit your `.env` file** with the required API keys:
   ```env
   DISCORD_BOT_TOKEN=your_discord_bot_token_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here  # Optional, for image generation
   TARGET_GUILD_ID=your_discord_server_id
   RESIDENCY_ROLE_NAME=25 residency
   ```

2. **Set up your Discord Bot**:
   - Go to https://discord.com/developers/applications
   - Create a new application and bot
   - Copy the bot token to your `.env` file
   - Enable necessary intents (Message Content Intent, Server Members Intent)

3. **Get your Guild ID**:
   - Enable Developer Mode in Discord
   - Right-click your server → Copy Server ID

### 3. Run the Bot

```bash
python discord_bot_auglab.py
```

## 💬 Usage

### Basic Interaction
- **Mention the bot**: `@AugLabBot Hello!`
- **Use prefix**: `!help` or `!budget`
- **Direct Message**: Send a DM to the bot
- **Smart Activation**: Bot responds to keywords like "auglab", "lab", "help"

### Commands

#### 📊 Budget Management
```
!budget                 # View detailed spending report
```

#### 🔍 Data Search
```
!search <query>         # Search stored channel history
!search AI research     # Example: Find messages about AI research
```

#### 🖼️ Image Generation
```
!generate_image <prompt>                    # Create images with DALL-E
!generate_image a futuristic laboratory     # Example usage
```

#### 👥 Member Management
```
!dm_residency <message>                     # Send DM to all residency members
!dm_residency Meeting at 3pm tomorrow       # Example usage
```

#### ❓ Help & Information
```
!help                   # Show all available commands and features
```

## 🧠 AI Intelligence

### Dual-Model Architecture

The bot uses a sophisticated dual-model approach:

1. **First Pass (Haiku)**: Quick assessment of message importance and complexity
2. **Second Pass (Sonnet 4)**: Deep processing for complex tasks requiring advanced reasoning

### Assessment Criteria

The bot evaluates messages on:
- **Technical Complexity**: Programming, research, analysis needs
- **Research Depth**: Knowledge-intensive questions
- **Creative Tasks**: Content generation, brainstorming
- **Code Assistance**: Debugging, implementation help
- **Community Impact**: Questions affecting the lab community

### Smart Budget Usage

- Automatically selects the most cost-effective model
- Tracks spending across all models and users
- Provides detailed cost breakdowns by model and time period
- Alerts when budget thresholds are reached (50%, 75%, 90%)

## 💾 Data Management

### Channel Data Storage
- **Complete Message History**: Stores all messages for future reference
- **Search Capabilities**: Full-text search across all stored data
- **Metadata Preservation**: Attachments, embeds, timestamps
- **Performance Optimized**: Indexed database for fast queries

### Member Cache
- **Role Tracking**: Monitors member roles and permissions
- **Activity Patterns**: Tracks engagement and participation
- **Residency Management**: Special handling for program participants

## 🛠️ MCP Tool Integration

### Available Tools

#### 🌐 Web Search
```
Format: [TOOL:web_search](query="search terms")
Example: Search for latest AI research papers
```

#### 📷 Image Analysis
```
Format: [TOOL:analyze_image](image_url="https://...")
Example: Analyze uploaded diagrams or charts
```

#### 💻 Code Execution
```
Format: [TOOL:execute_code](code="print('hello')")
Example: Run Python code safely
```

#### 📁 File Operations
```
Format: [TOOL:file_operations](operation="read", filepath="data.txt")
Example: Create, read, or modify files
```

## 📈 Budget & Cost Tracking

### Cost Structure (per 1M tokens)
- **Haiku Input**: $0.25
- **Haiku Output**: $1.25
- **Sonnet Input**: $3.00
- **Sonnet Output**: $15.00
- **GPT-4o**: $2.50 input, $10.00 output
- **DALL-E 3**: ~$0.04 per image

### Budget Monitoring
- **Real-time Tracking**: Every API call is logged with precise costs
- **Detailed Reports**: Breakdown by model, user, and time period
- **Automatic Alerts**: Notifications at 50%, 75%, and 90% budget usage
- **Emergency Stops**: Bot will refuse requests when budget is exhausted

## 🔧 Advanced Configuration

### Environment Variables

```env
# Core Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key

# Bot Behavior
BOT_PREFIX=!
COMPUTE_BUDGET=100.0
RESIDENCY_ROLE_NAME=25 residency
TARGET_GUILD_ID=your_server_id

# Model Selection
CLAUDE_MODEL_CHEAP=claude-3-haiku-20240307
CLAUDE_MODEL_EXPENSIVE=claude-3-5-sonnet-20241022

# Cost Rates (per 1M tokens)
HAIKU_INPUT_COST=0.25
HAIKU_OUTPUT_COST=1.25
SONNET_INPUT_COST=3.0
SONNET_OUTPUT_COST=15.0

# Database
DATABASE_PATH=./auglab_bot.db

# Features
MCP_ENABLE=True
DEBUG=False
```

### Database Management

The bot uses SQLite for data persistence:
- **Cost Entries**: All API usage and costs
- **Channel Data**: Complete message history
- **Member Cache**: User roles and metadata

Database files are automatically created and managed.

## 🔐 Security Features

### Safe Defaults
- **Input Validation**: All user inputs are sanitized
- **Path Traversal Protection**: File operations are restricted to safe directories
- **Rate Limiting**: Built-in delays for DM operations
- **Budget Enforcement**: Hard stops when budget is exhausted

### Privacy
- **Local Storage**: All data stored locally in SQLite
- **No External Sharing**: Channel data never leaves your server
- **User Consent**: Clear indication of data collection

## 🚀 Deployment

### Local Development
```bash
python discord_bot_auglab.py
```

### Production Deployment

1. **Using systemd** (recommended):
   ```bash
   sudo cp auglab-bot.service /etc/systemd/system/
   sudo systemctl enable auglab-bot
   sudo systemctl start auglab-bot
   ```

2. **Using Docker**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "discord_bot_auglab.py"]
   ```

3. **Using PM2**:
   ```bash
   npm install -g pm2
   pm2 start discord_bot_auglab.py --interpreter python3 --name auglab-bot
   ```

## 📊 Monitoring & Logs

### Built-in Monitoring
- **Budget Alerts**: Automatic notifications at spending thresholds
- **Error Logging**: Comprehensive error tracking and reporting
- **Performance Metrics**: Response times and token usage statistics

### External Monitoring
- **Health Checks**: Bot status monitoring
- **Log Aggregation**: Centralized logging for production environments
- **Metrics Export**: Prometheus-compatible metrics

## 🐛 Troubleshooting

### Common Issues

#### Bot Not Responding
1. Check bot permissions in Discord
2. Verify `DISCORD_BOT_TOKEN` is correct
3. Ensure Message Content Intent is enabled
4. Check bot has permission to read/send messages in channels

#### API Errors
1. Verify API keys in `.env` file
2. Check rate limits on Anthropic/OpenAI accounts
3. Ensure sufficient credits in API accounts

#### Database Issues
1. Check file permissions for database file
2. Verify disk space for SQLite operations
3. Run setup script to reinitialize database

#### Budget/Cost Issues
1. Check current spending with `!budget` command
2. Verify cost rates in environment variables
3. Review cost tracking logs

### Debug Mode
Enable detailed logging:
```env
DEBUG=True
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Install development dependencies
3. Run tests with `python -m pytest`
4. Submit pull requests for review

### Feature Requests
- Open GitHub issues for new feature requests
- Include detailed use cases and requirements
- Consider budget and security implications

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Review the logs for error messages
3. Open GitHub issues for bugs or feature requests
4. Contact the Augmentation Lab team for urgent issues

---

**Happy Augmenting! 🧪✨** 