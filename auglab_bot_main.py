#!/usr/bin/env python3
"""
Augmentation Lab Discord Bot - Main Entry Point
A sophisticated Discord bot with modular architecture
"""

import os
import logging
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO if not os.getenv('DEBUG', 'False').lower() == 'true' else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modular components
from auglab_bot.config import ConfigManager
from auglab_bot.storage import CostTracker, ChannelDataStore, MemorySystem, MessageStorage
from auglab_bot.management import WorkingMemoryManager
from auglab_bot.ai import ModelManager, FunctionCallManager, ResponseGenerator, ConversationManager
from auglab_bot.bot.functions import FunctionImplementations
from auglab_bot.utils import setup_log_capture

# Health check for Render deployment
try:
    from health_check import start_health_server
    HEALTH_CHECK_AVAILABLE = True
except ImportError:
    HEALTH_CHECK_AVAILABLE = False
    logger.warning("Health check server not available - optional for development")

import discord
from discord.ext import commands, tasks
import asyncio
from datetime import datetime
import json

class AugmentationLabBot(commands.Bot):
    """The main Augmentation Lab Discord bot with modular architecture."""
    
    def __init__(self):
        # Load configuration
        self.config = ConfigManager()
        
        # Set up log capture for debug command
        self.log_capture = setup_log_capture()
        logger.info("✅ Log capture initialized")
        
        # Basic configuration
        self.prefix = self.config.get('bot.prefix', '!')
        self.target_guild_id = self.config.get('bot.target_guild_id', 0)
        self.residency_role_name = self.config.get('bot.residency_role_name', "Resident '25")
        self.budget = self.config.get('budget.total_budget', 5.0)
        self.db_path = self.config.get('bot.database_path', './auglab_bot.db')
        self.allowed_guilds = self.config.get('bot.allowed_guilds', [])
        self.restricted_mode = self.config.get('bot.restricted_mode', False)
        self.response_mode = self.config.get('bot.response_mode', 'mentions_only')
        self.always_monitor_memory = self.config.get('bot.always_monitor_memory', True)
        
        # Initialize Discord bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True
        intents.members = True  # Required for find_member_by_name and member cache
        super().__init__(command_prefix=self.prefix, intents=intents, help_command=None)
        
        # Initialize storage components
        self.cost_tracker = CostTracker(self.budget, self.db_path)
        self.data_store = ChannelDataStore(self.db_path)
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.memory_system = MemorySystem(self.db_path, openai_api_key)
        self.message_storage = MessageStorage(self.db_path)
        
        # Initialize management components
        self.working_memory_manager = WorkingMemoryManager(self.db_path)
        
        # Initialize AI components
        self.model_manager = ModelManager(self.config)
        self.function_manager = FunctionCallManager()
        self.conversation_manager = ConversationManager(message_storage=self.message_storage)
        
        # Initialize function implementations
        self.function_implementations = FunctionImplementations(
            self.cost_tracker, self.memory_system, self.working_memory_manager,
            self.model_manager, self.config, self.data_store, self.db_path
        )
        # Set bot reference for functions that need it
        self.function_implementations.bot = self
        
        # Register functions with the function manager
        self.function_manager.register_functions(self.function_implementations.get_all_functions())
        
        # Initialize response generator
        self.response_generator = ResponseGenerator(
            self.model_manager, self.function_manager, self.conversation_manager,
            self.cost_tracker, self.memory_system, self.working_memory_manager, self.config
        )
        
        # Set references for autonomous message handling
        self.response_generator.bot = self
        self.response_generator.function_implementations = self.function_implementations
        
        # Background tasks
        
        logger.info("✅ Augmentation Lab Bot initialized with modular architecture")
    
    async def setup_hook(self):
        """Setup hook called when bot is ready."""
        logger.info("🔧 Setting up bot...")
        
        # Manually ensure commands are registered
        try:
            # Force command registration by checking if they exist
            command_methods = [
                ('budget', self.budget_command),
                ('test', self.test_command), 
                ('help', self.help_command),
                ('constitution', self.constitution_command),
                ('memory_query', self.memory_query_command),
                ('memory_stats', self.memory_stats_command)
            ]
            
            for cmd_name, cmd_func in command_methods:
                if hasattr(cmd_func, '__commands_is_command__'):
                    if not self.get_command(cmd_name):
                        self.add_command(cmd_func)
                        logger.info(f"✅ Manually registered command: {cmd_name}")
                    else:
                        logger.info(f"✅ Command already registered: {cmd_name}")
                        
            logger.info(f"📋 Total registered commands: {len(self.commands)}")
        except Exception as e:
            logger.error(f"❌ Error in command registration: {e}")
            
        logger.info("✅ Bot setup complete")
    
    async def on_ready(self):
        """Called when bot is ready."""
        logger.info(f"{self.user} has connected to Discord!")
        logger.info(f"Bot is in {len(self.guilds)} guilds")
        logger.info(f"Budget remaining: ${self.cost_tracker.get_remaining_budget():.2f}")
        
        # Update conversation manager with bot user ID for proper DM handling
        self.conversation_manager.bot_user_id = self.user.id
        logger.info(f"🤖 Updated ConversationManager with bot user ID: {self.user.id}")
        
        # Start health server  
        if HEALTH_CHECK_AVAILABLE:
            await start_health_server()
            logger.info("✅ Health check server started")
        
        # Handle database migration for Render deployment
        await self.handle_database_migration()
        
        # Update member cache
        await self.update_member_cache()
        
        # Enable semantic search
        await self.enable_semantic_search()
        
        # Check if memory migration is needed
        try:
            stats = self.memory_system.get_memory_stats()
            total_memories = stats.get('total_memories', 0)
            with_embeddings = stats.get('memories_with_embeddings', 0)
            
            if total_memories > 0:
                logger.info(f"📊 Memory Stats: {total_memories} total, {with_embeddings} with embeddings")
                logger.info(f"📊 Coverage: {stats.get('embedding_coverage', 'unknown')}")
                logger.info(f"📊 Types: {stats.get('memory_types', {})}")
                
                if with_embeddings < total_memories:
                    missing = total_memories - with_embeddings
                    logger.info(f"🔄 Regenerating embeddings for {missing} memories (processing 50 at a time)...")
                    updated = self.memory_system.regenerate_embeddings_for_existing_memories(50)
                    logger.info(f"✅ Regenerated embeddings for {updated} memories!")
                else:
                    logger.info("✅ All memories have embeddings - semantic search fully operational")
            else:
                logger.info("📝 No memories found - starting fresh")
                
        except Exception as e:
            logger.error(f"Embedding regeneration error: {e}")
    
    async def handle_database_migration(self):
        """Handle database migration from repository to persistent disk on Render."""
        try:
            import os
            import shutil
            
            # Check if we're on Render (has persistent disk path)
            persistent_db_path = os.getenv('DATABASE_PATH', self.db_path)
            repo_db_path = './auglab_bot.db'  # Database from repository
            
            # If DATABASE_PATH is set (Render environment) and repo database exists
            if persistent_db_path != self.db_path and os.path.exists(repo_db_path):
                # Ensure persistent disk directory exists
                persistent_dir = os.path.dirname(persistent_db_path)
                os.makedirs(persistent_dir, exist_ok=True)
                
                # Copy repository database to persistent disk if not already there
                if not os.path.exists(persistent_db_path):
                    logger.info(f"📦 Copying database from repository to persistent disk...")
                    logger.info(f"   From: {repo_db_path}")
                    logger.info(f"   To: {persistent_db_path}")
                    
                    shutil.copy2(repo_db_path, persistent_db_path)
                    logger.info("✅ Database migration to persistent disk completed!")
                    
                    # Update the bot to use persistent database
                    self.db_path = persistent_db_path
                    
                    # Reinitialize components with new database path
                    self.cost_tracker = CostTracker(self.budget, self.db_path)
                    self.data_store = ChannelDataStore(self.db_path)
                    openai_api_key = os.getenv('OPENAI_API_KEY')
                    self.memory_system = MemorySystem(self.db_path, openai_api_key)
                    
                    logger.info(f"✅ Bot components updated to use persistent database: {self.db_path}")
                else:
                    logger.info(f"✅ Persistent database already exists: {persistent_db_path}")
                    # Still update to use persistent path
                    self.db_path = persistent_db_path
                    
        except Exception as e:
            logger.warning(f"Database migration note: {e} (may be expected in local development)")
    
    async def enable_semantic_search(self):
        """Enable semantic search for memory system."""
        try:
            await self.memory_system.enable_semantic_search()
            logger.info("✅ Semantic search enabled with OpenAI embeddings")
        except Exception as e:
            logger.error(f"Failed to enable semantic search: {e}")
    
    async def update_member_cache(self):
        """Update member cache for all guilds."""
        try:
            total_members = 0
            for guild in self.guilds:
                try:
                    members = await guild.chunk()
                    total_members += len(members)
                    
                    # Store member info
                    for member in members:
                        self.data_store.store_member(member)
                except Exception as e:
                    if "Intents.members must be enabled" in str(e):
                        logger.warning("⚠️ Members intent not enabled - skipping member cache update")
                        return
                    else:
                        # Log the actual exception type for debugging
                        logger.error(f"Unexpected error in member cache update: {type(e).__name__}: {e}")
                        raise
            
            logger.info(f"Updated member cache for {total_members} members")
        except Exception as e:
            logger.error(f"Error updating member cache: {e}")
    
    async def on_message(self, message):
        """Handle all incoming messages."""
        # Don't respond to bots
        if message.author.bot:
            return
        
        # Check guild restrictions
        if self.restricted_mode and message.guild:
            if self.allowed_guilds and message.guild.id not in self.allowed_guilds:
                return
        
        # Check residency role
        if message.guild:
            member = message.guild.get_member(message.author.id)
            if member:
                role_exists = any(role.name == self.residency_role_name for role in message.guild.roles)
                if role_exists:
                    if not any(role.name == self.residency_role_name for role in member.roles):
                        return
        
        # Store message in data store
        self.data_store.store_message(message)
        
        # Store message for 24h conversation context
        self.message_storage.store_message(message)
        
        # Process commands first
        await self.process_commands(message)
        
        # Check if valid command
        if message.content.startswith(self.prefix):
            parts = message.content[len(self.prefix):].split()
            if parts:
                command_name = parts[0].lower()
                valid_commands = ['budget', 'constitution', 'memory_query', 'user_history', 
                                'memory_stats', 'dm_residency', 'search', 'generate_image', 'help',
                                'autonomous_think', 'autonomous_status', 'guilds', 'models', 'switch_model',
                                'response_mode', 'debug', 'status']
                if command_name in valid_commands:
                    return
        
        # Determine response behavior
        should_monitor_memory = self.always_monitor_memory and (
            isinstance(message.channel, discord.DMChannel) or
            (message.guild and self.target_guild_id and message.guild.id == self.target_guild_id) or
            (message.guild and not self.target_guild_id) or
            any(word in message.content.lower() for word in ['auglab', 'lab', 'project'])
        )
        
        should_respond = self._should_respond_to_message(message)
        
        # Monitor memory if enabled
        if should_monitor_memory:
            await self.memory_monitor(message)
        
        # Generate response if needed
        if should_respond:
            await self.handle_ai_response(message)
    
    def _should_respond_to_message(self, message) -> bool:
        """Determine if bot should respond to a message."""
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.user in message.mentions
        
        logger.info(f"🎯 Checking response criteria: mode={self.response_mode}, is_dm={is_dm}, is_mentioned={is_mentioned}")
        
        if self.response_mode == 'mentions_only':
            should_respond = is_mentioned or is_dm
        elif self.response_mode == 'all_messages':
            should_respond = True
        elif self.response_mode == 'dms_only':
            should_respond = is_dm
        else:  # disabled
            should_respond = False
        
        if is_dm:
            logger.info(f"🎯 DM detected: should_respond={should_respond} (mode allows DMs: {self.response_mode in ['mentions_only', 'all_messages', 'dms_only']})")
        
        return should_respond
    
    async def memory_monitor(self, message):
        """Monitor messages for memory storage."""
        try:
            content = message.content
            if not content.strip():
                return
            
            # Use cheap model to assess memory importance
            assessment_prompt = f"""Analyze this message for memory storage:

Message: "{content}"
Author: {message.author.display_name}
Channel: #{message.channel.name if hasattr(message.channel, 'name') else 'DM'}

Rate importance (1-10) and decide if worth storing. Consider:
- Strategic value for Aug Lab
- Useful information for future reference
- Technical insights or solutions
- Social connections and relationships

Respond with JSON: {{"importance": X, "store": true/false, "tags": ["tag1", "tag2"], "summary": "brief summary"}}"""

            cheap_config = self.model_manager.get_model_config('cheap')
            response = await self.model_manager.call_model_api(cheap_config, assessment_prompt, 150)
            
            try:
                assessment = json.loads(response.strip())
                if assessment.get('store', False) and assessment.get('importance', 0) >= 4:
                    # Store in memory
                    memory_id = self.memory_system.store_memory(
                        content,
                        message.author.id,
                        message.channel.id,
                        memory_type='conversation',
                        tags=assessment.get('tags', [])
                    )
                    logger.info(f"🧠 Auto-stored memory (importance: {assessment.get('importance')}): {content[:50]}...")
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse memory assessment response")
                
        except Exception as e:
            logger.error(f"Error in memory monitoring: {e}")
    
    async def handle_ai_response(self, message):
        """Handle AI response generation."""
        try:
            # Show typing indicator immediately to let user know bot is responding
            async with message.channel.typing():
                response = await self.response_generator.handle_ai_response(message)
                
                if response:
                    # Split long responses
                    if len(response) > 2000:
                        parts = [response[i:i+2000] for i in range(0, len(response), 2000)]
                        for part in parts:
                            await message.reply(part)
                    else:
                        await message.reply(response)
                
                # Process any pending autonomous messages
                await self.process_pending_autonomous_messages()
                    
        except Exception as e:
            logger.error(f"Error in AI response: {e}")
            await message.reply("Sorry, I encountered an error while processing your request.")
    
    async def process_pending_autonomous_messages(self):
        """Process and send any pending autonomous messages."""
        try:
            if hasattr(self.function_implementations, '_pending_autonomous_message') and self.function_implementations._pending_autonomous_message:
                pending = self.function_implementations._pending_autonomous_message
                channel = pending['channel']
                message_content = pending['message']
                
                logger.info(f"📤 Sending autonomous message to #{channel.name}: {message_content[:100]}...")
                
                # Send the message
                await channel.send(message_content)
                
                # Clear the pending message
                self.function_implementations._pending_autonomous_message = None
                
                logger.info(f"✅ Autonomous message sent successfully to #{channel.name}")
                
        except Exception as e:
            logger.error(f"Error sending autonomous message: {e}")
            # Clear the pending message even on error to prevent loops
            if hasattr(self.function_implementations, '_pending_autonomous_message'):
                self.function_implementations._pending_autonomous_message = None
    

    
    # Commands
    @commands.command(name='budget')
    async def budget_command(self, ctx):
        """Show budget information."""
        try:
            report = self.cost_tracker.get_spending_report()
            percentage_used = (report['total_spent'] / report['budget']) * 100
            
            embed = discord.Embed(
                title="💰 Budget Status",
                description=f"Budget management for compute costs",
                color=0x00ff00 if percentage_used < 75 else 0xffaa00 if percentage_used < 90 else 0xff0000
            )
            
            embed.add_field(
                name="💵 Total Budget",
                value=f"${report['budget']:.2f}",
                inline=True
            )
            
            embed.add_field(
                name="💸 Spent",
                value=f"${report['total_spent']:.4f}",
                inline=True
            )
            
            embed.add_field(
                name="💳 Remaining",
                value=f"${report['remaining']:.2f}",
                inline=True
            )
            
            embed.add_field(
                name="📊 Usage",
                value=f"{percentage_used:.1f}%",
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"Error getting budget info: {e}")

    @commands.command(name='test')
    async def test_command(self, ctx):
        """Simple test command."""
        await ctx.send("✅ Test command working!")

    @commands.command(name='help')
    async def help_command(self, ctx):
        """Show help information."""
        embed = discord.Embed(
            title="🧪 Augmentation Lab Bot Help",
            description="Available commands and features",
            color=0x0099ff
        )
        
        embed.add_field(
            name="💰 Budget Commands", 
            value="`!budget` - Show budget status",
            inline=False
        )
        
        embed.add_field(
            name="🧠 Memory Commands",
            value="`!memory_query <query>` - Search memories\n`!memory_stats` - Show memory statistics",
            inline=False
        )
        
        embed.add_field(
            name="📜 Constitution Commands",
            value="`!constitution` - View current constitution",
            inline=False
        )
        
        embed.add_field(
            name="🤖 AI Features",
            value="• Dual-model system (Haiku/Sonnet)\n• Function calling\n• Long-term memory\n• Autonomous mode",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='constitution')
    async def constitution_command(self, ctx):
        """Show the current constitution."""
        try:
            constitution = self.constitution_manager.get_current_constitution()
            
            if not constitution:
                await ctx.send("❌ No constitution found in database.")
                return
            
            if len(constitution) > 1900:  # Discord embed limit
                # Split into chunks
                chunks = [constitution[i:i+1900] for i in range(0, len(constitution), 1900)]
                await ctx.send("📜 **Current Constitution** (Part 1):")
                await ctx.send(f"```\n{chunks[0]}\n```")
                
                for i, chunk in enumerate(chunks[1:], 2):
                    await ctx.send(f"📜 **Constitution** (Part {i}):")
                    await ctx.send(f"```\n{chunk}\n```")
            else:
                embed = discord.Embed(
                    title="📜 Current Constitution",
                    description=f"```\n{constitution}\n```",
                    color=discord.Color.blue()
                )
                await ctx.send(embed=embed)
                
        except Exception as e:
            logger.error(f"Error getting constitution: {e}")
            await ctx.send(f"❌ Error getting constitution: {e}")
    
    @commands.command(name='memory_query')
    async def memory_query_command(self, ctx, *, query: str):
        """Search memories."""
        try:
            memories = self.memory_system.query_memories(query, limit=5)
            
            if not memories:
                await ctx.send(f"🔍 No memories found for: `{query}`")
                return
            
            embed = discord.Embed(
                title="🧠 Memory Search Results",
                description=f"Query: `{query}`",
                color=0x0099ff
            )
            
            for i, memory in enumerate(memories[:3], 1):  # Show top 3
                content = memory['content'][:200] + "..." if len(memory['content']) > 200 else memory['content']
                timestamp = memory.get('created_at', 'Unknown')
                memory_type = memory.get('memory_type', 'unknown')
                
                embed.add_field(
                    name=f"📋 Result {i} ({memory_type})",
                    value=f"**Time:** {timestamp}\n**Content:** {content}",
                    inline=False
                )
            
            if len(memories) > 3:
                embed.set_footer(text=f"Showing 3 of {len(memories)} results")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            await ctx.send(f"❌ Error searching memories: {e}")
    
    @commands.command(name='memory_stats')
    async def memory_stats_command(self, ctx):
        """Show memory system statistics."""
        try:
            # Get memory stats (implement this method in MemorySystem if needed)
            stats = {
                "total_memories": "Unknown",
                "unique_users": "Unknown", 
                "memory_types": "Unknown"
            }
            
            embed = discord.Embed(
                title="🧠 Memory System Statistics",
                color=0x0099ff
            )
            
            embed.add_field(name="📊 Total Memories", value=stats["total_memories"], inline=True)
            embed.add_field(name="👥 Unique Users", value=stats["unique_users"], inline=True)
            embed.add_field(name="📂 Memory Types", value=stats["memory_types"], inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            await ctx.send(f"❌ Error getting memory stats: {e}")

    @tasks.loop(hours=6)
    async def cleanup_old_messages(self):
        """Clean up old messages to prevent database bloat."""
        try:
            deleted_count = self.message_storage.cleanup_old_messages(hours=48)
            if deleted_count > 0:
                logger.info(f"🧹 Message cleanup: removed {deleted_count} old messages")
        except Exception as e:
            logger.error(f"❌ Error in message cleanup: {e}")

    async def setup_hook(self):
        """Setup background tasks when bot is ready."""
        if not self.cleanup_old_messages.is_running():
            self.cleanup_old_messages.start()
            logger.info("🚀 Started message cleanup background task")


def main():
    """Main function to run the bot."""
    # Check for required environment variables
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    if not bot_token:
        logger.error("DISCORD_BOT_TOKEN environment variable is required")
        return
    
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_key:
        logger.error("ANTHROPIC_API_KEY environment variable is required")
        return
    
    try:
        bot = AugmentationLabBot()
        logger.info("Starting Augmentation Lab Discord Bot...")
        logger.info(f"Budget: ${bot.budget}")
        logger.info(f"Target Guild ID: {bot.target_guild_id}")
        logger.info(f"Database: {bot.db_path}")
        
        bot.run(bot_token)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main() 