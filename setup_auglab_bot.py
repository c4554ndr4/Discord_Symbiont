#!/usr/bin/env python3
"""
Setup Script for Augmentation Lab Discord Bot
Helps configure and initialize the bot with proper settings.
"""

import os
import sqlite3
import shutil
from pathlib import Path

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_path = Path('.env')
    env_example_path = Path('env_example.txt')
    
    if not env_path.exists() and env_example_path.exists():
        shutil.copy(env_example_path, env_path)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your actual API keys and configuration")
        return True
    elif env_path.exists():
        print("ℹ️  .env file already exists")
        return False
    else:
        print("❌ env_example.txt not found")
        return False

def initialize_database():
    """Initialize the SQLite database."""
    db_path = os.getenv('DATABASE_PATH', './auglab_bot.db')
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Cost tracking table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cost_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    action TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL
                )
            ''')
            
            # Channel data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS channel_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER NOT NULL,
                    channel_name TEXT NOT NULL,
                    guild_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    author_id INTEGER NOT NULL,
                    author_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    attachments TEXT,
                    embeds TEXT,
                    UNIQUE(message_id)
                )
            ''')
            
            # Member cache table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS member_cache (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    roles TEXT NOT NULL,
                    joined_at TEXT,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # Constitution table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS constitution (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1
                )
            ''')
            
            # Memory system tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    embedding_summary TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memory_connections (
                    from_memory_id TEXT NOT NULL,
                    to_memory_id TEXT NOT NULL,
                    connection_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (from_memory_id, to_memory_id)
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_channel_data_channel_id ON channel_data(channel_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_channel_data_timestamp ON channel_data(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cost_entries_timestamp ON cost_entries(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cost_entries_user_id ON cost_entries(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_connections_from ON memory_connections(from_memory_id)')
            
        print(f"✅ Database initialized at {db_path}")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    import_map = {
        'discord.py': 'discord',
        'anthropic': 'anthropic',
        'openai': 'openai',
        'python-dotenv': 'dotenv',
        'requests': 'requests',
        'tiktoken': 'tiktoken'
    }
    
    missing_packages = []
    
    for package, import_name in import_map.items():
        try:
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def validate_env_config():
    """Validate environment configuration."""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        'DISCORD_BOT_TOKEN',
        'ANTHROPIC_API_KEY'
    ]
    
    optional_vars = [
        'OPENAI_API_KEY',
        'TARGET_GUILD_ID',
        'RESIDENCY_ROLE_NAME'
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        print(f"❌ Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional environment variables: {', '.join(missing_optional)}")
        print("   Some features may not work without these")
    
    print("✅ Environment configuration looks good")
    return True

def create_systemd_service():
    """Create a systemd service file for running the bot."""
    service_content = f"""[Unit]
Description=Augmentation Lab Discord Bot
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'ubuntu')}
WorkingDirectory={os.getcwd()}
ExecStart={os.sys.executable} discord_bot_auglab.py
Restart=always
RestartSec=5
Environment=PATH={os.environ.get('PATH')}

[Install]
WantedBy=multi-user.target
"""
    
    service_path = Path('auglab-bot.service')
    with open(service_path, 'w') as f:
        f.write(service_content)
    
    print(f"✅ Created systemd service file: {service_path}")
    print("💡 To install: sudo cp auglab-bot.service /etc/systemd/system/")
    print("💡 To enable: sudo systemctl enable auglab-bot")
    print("💡 To start: sudo systemctl start auglab-bot")

def main():
    """Main setup function."""
    print("🧪 Augmentation Lab Discord Bot Setup")
    print("=" * 40)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Setup failed - missing dependencies")
        return
    
    # Create environment file
    env_created = create_env_file()
    
    # Validate configuration
    if not validate_env_config():
        if env_created:
            print(f"\n💡 Please edit .env file with your configuration and run setup again")
        return
    
    # Initialize database
    if not initialize_database():
        print("\n❌ Setup failed - database initialization error")
        return
    
    # Create systemd service
    create_systemd_service()
    
    print("\n" + "=" * 40)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Review and update .env file with your API keys")
    print("2. Set TARGET_GUILD_ID to your Discord server ID")
    print("3. Set RESIDENCY_ROLE_NAME to your residency role name")
    print("4. Run the bot: python discord_bot_auglab.py")
    print("\n🔧 Optional:")
    print("- Install systemd service for auto-restart")
    print("- Set up log rotation")
    print("- Configure firewall if needed")

if __name__ == "__main__":
    main() 