#!/usr/bin/env python3
"""
Configuration helper for Augmentation Lab Discord Bot.
This script helps you set up the .env file with your existing API keys.
"""

import os
import shutil

def create_env_file():
    """Create .env file from template."""
    print("🧪 Augmentation Lab Bot Configuration")
    print("=" * 40)
    
    # Check if .env already exists
    if os.path.exists('.env'):
        overwrite = input("⚠️  .env file already exists. Overwrite? (y/N): ").lower()
        if overwrite != 'y':
            print("Cancelled.")
            return
    
    # Copy from example
    if os.path.exists('env_example.txt'):
        shutil.copy('env_example.txt', '.env')
        print("✅ Created .env from template")
    else:
        print("❌ env_example.txt not found")
        return
    
    print("\n📝 Please edit .env file with your actual values:")
    print("\n1. DISCORD_BOT_TOKEN - Get from Discord Developer Portal")
    print("   → https://discord.com/developers/applications")
    print("   → Create Application → Bot → Copy Token")
    
    print("\n2. ANTHROPIC_API_KEY - Get from Anthropic Console")
    print("   → https://console.anthropic.com/")
    print("   → API Keys → Create Key")
    
    print("\n3. OPENAI_API_KEY - Get from OpenAI Platform")
    print("   → https://platform.openai.com/api-keys")
    print("   → Create new secret key")
    
    print("\n4. TARGET_GUILD_ID - Your Discord Server ID")
    print("   → Enable Developer Mode in Discord")
    print("   → Right-click your server → Copy Server ID")
    
    print("\n5. RESIDENCY_ROLE_NAME - Name of your residency role")
    print("   → Default: '25 residency'")
    
    print("\n📁 File location: .env")
    print("💡 After editing, run: python discord_bot_auglab.py")

def check_current_config():
    """Check current configuration status."""
    print("🔍 Current Configuration Status")
    print("=" * 35)
    
    if not os.path.exists('.env'):
        print("❌ No .env file found")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    configs = [
        ('DISCORD_BOT_TOKEN', 'Discord Bot Token'),
        ('ANTHROPIC_API_KEY', 'Anthropic API Key'),
        ('OPENAI_API_KEY', 'OpenAI API Key'),
        ('TARGET_GUILD_ID', 'Guild ID'),
        ('RESIDENCY_ROLE_NAME', 'Residency Role')
    ]
    
    all_good = True
    
    for env_var, description in configs:
        value = os.getenv(env_var)
        if value and not value.startswith('your_'):
            print(f"✅ {description}: {'*' * 10}...{value[-4:]}")
        else:
            print(f"❌ {description}: Not configured")
            all_good = False
    
    return all_good

def test_discord_connection():
    """Test if Discord token works."""
    print("\n🔗 Testing Discord Connection...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token or token.startswith('your_'):
        print("❌ Discord token not configured")
        return False
    
    try:
        import discord
        import asyncio
        
        async def test_login():
            try:
                client = discord.Client(intents=discord.Intents.default())
                await client.login(token)
                await client.close()
                return True
            except Exception as e:
                print(f"❌ Discord login failed: {e}")
                return False
        
        result = asyncio.run(test_login())
        if result:
            print("✅ Discord token is valid")
        return result
        
    except Exception as e:
        print(f"❌ Error testing Discord: {e}")
        return False

def main():
    """Main configuration flow."""
    print("Welcome to the Augmentation Lab Bot setup!\n")
    
    choice = input("Choose an option:\n1. Create new .env file\n2. Check current config\n3. Test Discord connection\n\nEnter choice (1-3): ")
    
    if choice == '1':
        create_env_file()
    elif choice == '2':
        if check_current_config():
            print("\n✅ Configuration looks complete!")
        else:
            print("\n⚠️  Some configuration items need attention.")
    elif choice == '3':
        if check_current_config():
            test_discord_connection()
        else:
            print("❌ Please configure .env file first")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 