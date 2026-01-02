#!/bin/bash
echo "🧪 Starting Augmentation Lab Discord Bot..."

# Kill any existing bot instances
echo "🔄 Checking for existing bot instances..."
if pgrep -f "auglab_bot_main.py" > /dev/null; then
    echo "⚠️ Found existing bot instances, terminating them..."
    pkill -f "auglab_bot_main.py"
    sleep 3
    echo "✅ Existing instances terminated"
else
    echo "✅ No existing instances found"
fi

# Also check for legacy instances
if pgrep -f "discord_bot_auglab.py" > /dev/null; then
    echo "⚠️ Found legacy bot instances, terminating them..."
    pkill -f "discord_bot_auglab.py"
    sleep 2
    echo "✅ Legacy instances terminated"
fi

echo "🔍 Checking configuration..."

if grep -q "YOUR_DISCORD_BOT_TOKEN_HERE" .env; then
    echo "❌ Please edit .env file and set your Discord bot token"
    echo "   Get token from: https://discord.com/developers/applications"
    exit 1
fi

if grep -q "YOUR_ANTHROPIC_API_KEY_HERE" .env; then
    echo "❌ Please edit .env file and set your Anthropic API key"
    echo "   Get key from: https://console.anthropic.com/"
    exit 1
fi

echo "✅ Configuration looks good!"
echo "🚀 Starting bot..."
python discord_bot_auglab.py
