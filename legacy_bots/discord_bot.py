#!/usr/bin/env python3
"""
Discord Bot with Claude Opus Integration
A context-aware Discord bot that uses Anthropic's Claude Opus for intelligent responses.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

import discord
from discord.ext import commands
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO if not os.getenv('DEBUG', 'False').lower() == 'true' else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContextAwareBot(commands.Bot):
    """A Discord bot with full context awareness and Claude integration."""
    
    def __init__(self):
        # Bot configuration
        self.prefix = os.getenv('BOT_PREFIX', '!')
        self.max_history = int(os.getenv('MAX_MESSAGE_HISTORY', '50'))
        self.claude_model = os.getenv('CLAUDE_MODEL', 'claude-3-opus-20240229')
        
        # Allowed channels (if specified)
        allowed_channels_str = os.getenv('ALLOWED_CHANNELS', '')
        self.allowed_channels = [int(ch.strip()) for ch in allowed_channels_str.split(',') if ch.strip()]
        
        # Initialize Discord bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        intents.guild_messages = True
        intents.dm_messages = True
        
        super().__init__(
            command_prefix=self.prefix,
            intents=intents,
            help_command=None  # We'll create a custom help command
        )
        
        # Initialize Anthropic client
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        
        # Context storage
        self.conversation_history: Dict[int, List[Dict[str, Any]]] = {}
        self.server_context: Dict[int, Dict[str, Any]] = {}
        self.user_context: Dict[int, Dict[str, Any]] = {}
        
    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        
        # Initialize server contexts
        for guild in self.guilds:
            await self.update_server_context(guild)
        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=f"{self.prefix}help | Claude Opus AI"
            )
        )
    
    async def on_guild_join(self, guild):
        """Called when the bot joins a new guild."""
        logger.info(f'Joined new guild: {guild.name} (ID: {guild.id})')
        await self.update_server_context(guild)
    
    async def update_server_context(self, guild):
        """Update context information for a server."""
        try:
            # Get server information
            server_info = {
                'name': guild.name,
                'id': guild.id,
                'member_count': guild.member_count,
                'created_at': guild.created_at.isoformat(),
                'owner_id': guild.owner_id,
                'channels': [],
                'roles': [],
                'categories': []
            }
            
            # Get channel information
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    server_info['channels'].append({
                        'name': channel.name,
                        'id': channel.id,
                        'category': channel.category.name if channel.category else None,
                        'topic': channel.topic
                    })
                elif isinstance(channel, discord.CategoryChannel):
                    server_info['categories'].append({
                        'name': channel.name,
                        'id': channel.id
                    })
            
            # Get role information
            for role in guild.roles:
                if role.name != '@everyone':
                    server_info['roles'].append({
                        'name': role.name,
                        'id': role.id,
                        'color': str(role.color),
                        'permissions': role.permissions.value
                    })
            
            self.server_context[guild.id] = server_info
            logger.debug(f'Updated context for guild: {guild.name}')
            
        except Exception as e:
            logger.error(f'Error updating server context for {guild.name}: {e}')
    
    async def update_user_context(self, user, guild=None):
        """Update context information for a user."""
        try:
            user_info = {
                'username': user.name,
                'display_name': user.display_name,
                'id': user.id,
                'created_at': user.created_at.isoformat(),
                'avatar_url': str(user.avatar.url) if user.avatar else None,
                'bot': user.bot
            }
            
            # Add guild-specific information if available
            if guild and isinstance(user, discord.Member):
                user_info.update({
                    'nickname': user.nick,
                    'joined_at': user.joined_at.isoformat() if user.joined_at else None,
                    'roles': [role.name for role in user.roles if role.name != '@everyone'],
                    'top_role': user.top_role.name if user.top_role.name != '@everyone' else None,
                    'permissions': user.guild_permissions.value
                })
            
            self.user_context[user.id] = user_info
            logger.debug(f'Updated context for user: {user.name}')
            
        except Exception as e:
            logger.error(f'Error updating user context for {user.name}: {e}')
    
    def get_conversation_context(self, channel_id: int) -> str:
        """Get formatted conversation context for Claude."""
        if channel_id not in self.conversation_history:
            return "No previous conversation history."
        
        history = self.conversation_history[channel_id]
        if not history:
            return "No previous conversation history."
        
        context_lines = []
        for msg in history[-10:]:  # Last 10 messages for context
            timestamp = msg['timestamp']
            author = msg['author']
            content = msg['content'][:200]  # Truncate long messages
            context_lines.append(f"[{timestamp}] {author}: {content}")
        
        return "\n".join(context_lines)
    
    def get_server_context_summary(self, guild_id: int) -> str:
        """Get a summary of server context for Claude."""
        if guild_id not in self.server_context:
            return "No server context available."
        
        server = self.server_context[guild_id]
        
        summary = f"""Server: {server['name']} (ID: {server['id']})
Members: {server['member_count']}
Channels: {len(server['channels'])} text channels
Categories: {', '.join([cat['name'] for cat in server['categories']])}
Key Roles: {', '.join([role['name'] for role in server['roles'][:5]])}"""
        
        return summary
    
    def get_user_context_summary(self, user_id: int) -> str:
        """Get a summary of user context for Claude."""
        if user_id not in self.user_context:
            return "No user context available."
        
        user = self.user_context[user_id]
        
        summary = f"""User: {user['display_name']} (@{user['username']})
Account created: {user['created_at'][:10]}"""
        
        if 'roles' in user and user['roles']:
            summary += f"\nRoles: {', '.join(user['roles'])}"
        
        if 'joined_at' in user and user['joined_at']:
            summary += f"\nJoined server: {user['joined_at'][:10]}"
        
        return summary
    
    async def add_to_conversation_history(self, message):
        """Add a message to conversation history."""
        channel_id = message.channel.id
        
        if channel_id not in self.conversation_history:
            self.conversation_history[channel_id] = []
        
        # Add message to history
        msg_data = {
            'timestamp': message.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'author': message.author.display_name,
            'content': message.content,
            'message_id': message.id,
            'user_id': message.author.id
        }
        
        self.conversation_history[channel_id].append(msg_data)
        
        # Keep only recent messages
        if len(self.conversation_history[channel_id]) > self.max_history:
            self.conversation_history[channel_id] = self.conversation_history[channel_id][-self.max_history:]
    
    async def generate_claude_response(self, message, user_prompt: str) -> str:
        """Generate a response using Claude with full context."""
        try:
            # Update contexts
            await self.update_user_context(message.author, message.guild)
            if message.guild:
                await self.update_server_context(message.guild)
            
            # Build context for Claude
            context_parts = []
            
            # Server context
            if message.guild:
                server_context = self.get_server_context_summary(message.guild.id)
                context_parts.append(f"SERVER CONTEXT:\n{server_context}")
            
            # User context
            user_context = self.get_user_context_summary(message.author.id)
            context_parts.append(f"USER CONTEXT:\n{user_context}")
            
            # Conversation history
            conversation_context = self.get_conversation_context(message.channel.id)
            context_parts.append(f"RECENT CONVERSATION:\n{conversation_context}")
            
            # Channel context
            channel_info = f"CURRENT CHANNEL: #{message.channel.name}"
            if hasattr(message.channel, 'topic') and message.channel.topic:
                channel_info += f" - {message.channel.topic}"
            context_parts.append(channel_info)
            
            # Build the full prompt
            full_context = "\n\n".join(context_parts)
            
            system_prompt = f"""You are Claude, an AI assistant integrated into a Discord server. You have full context about the server, users, and conversation history.

{full_context}

Please respond naturally and helpfully to the user's message. You can reference:
- Server information (channels, roles, members)
- User information (roles, join date, etc.)
- Previous conversation context
- Current channel context

Be conversational, helpful, and engaging. You're part of this Discord community."""

            # Make API call to Claude
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f'Error generating Claude response: {e}')
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    async def on_message(self, message):
        """Handle incoming messages."""
        # Don't respond to bot messages
        if message.author.bot:
            return
        
        # Add to conversation history
        await self.add_to_conversation_history(message)
        
        # Check if message is in allowed channels (if specified)
        if self.allowed_channels and message.channel.id not in self.allowed_channels:
            return
        
        # Check if bot is mentioned or message starts with prefix
        bot_mentioned = self.user in message.mentions
        starts_with_prefix = message.content.startswith(self.prefix)
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        if bot_mentioned or starts_with_prefix or is_dm:
            # Remove mention and prefix from message
            content = message.content
            if bot_mentioned:
                content = content.replace(f'<@{self.user.id}>', '').strip()
            if starts_with_prefix:
                content = content[len(self.prefix):].strip()
            
            if not content:
                content = "Hello! How can I help you?"
            
            # Show typing indicator
            async with message.channel.typing():
                response = await self.generate_claude_response(message, content)
            
            # Split long responses
            if len(response) > 2000:
                chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                for chunk in chunks:
                    await message.reply(chunk)
            else:
                await message.reply(response)
        
        # Process commands
        await self.process_commands(message)

# Custom commands
@commands.command(name='help')
async def help_command(ctx):
    """Show help information."""
    embed = discord.Embed(
        title="Claude Discord Bot Help",
        description="I'm Claude, an AI assistant with full context awareness!",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="How to interact with me:",
        value=f"• Mention me: @{ctx.bot.user.name}\n• Use prefix: `{ctx.bot.prefix}<message>`\n• Send me a DM",
        inline=False
    )
    
    embed.add_field(
        name="Commands:",
        value=f"`{ctx.bot.prefix}help` - Show this help\n`{ctx.bot.prefix}info` - Bot information\n`{ctx.bot.prefix}context` - Show current context",
        inline=False
    )
    
    embed.add_field(
        name="Context Awareness:",
        value="I can see:\n• Server info (channels, roles, members)\n• User info (roles, join dates)\n• Conversation history\n• Channel topics and categories",
        inline=False
    )
    
    await ctx.send(embed=embed)

@commands.command(name='info')
async def info_command(ctx):
    """Show bot information."""
    embed = discord.Embed(
        title="Bot Information",
        color=discord.Color.green()
    )
    
    embed.add_field(name="Model", value=ctx.bot.claude_model, inline=True)
    embed.add_field(name="Servers", value=len(ctx.bot.guilds), inline=True)
    embed.add_field(name="Prefix", value=ctx.bot.prefix, inline=True)
    
    embed.add_field(
        name="Features",
        value="• Claude Opus AI\n• Full context awareness\n• Conversation memory\n• Server & user context",
        inline=False
    )
    
    await ctx.send(embed=embed)

@commands.command(name='context')
async def context_command(ctx):
    """Show current context information."""
    embed = discord.Embed(
        title="Current Context",
        color=discord.Color.orange()
    )
    
    # Server context
    if ctx.guild:
        server_context = ctx.bot.get_server_context_summary(ctx.guild.id)
        embed.add_field(name="Server Context", value=f"```{server_context}```", inline=False)
    
    # User context
    user_context = ctx.bot.get_user_context_summary(ctx.author.id)
    embed.add_field(name="Your Context", value=f"```{user_context}```", inline=False)
    
    # Conversation history count
    channel_id = ctx.channel.id
    history_count = len(ctx.bot.conversation_history.get(channel_id, []))
    embed.add_field(name="Conversation History", value=f"{history_count} messages stored", inline=True)
    
    await ctx.send(embed=embed)

def main():
    """Main function to run the bot."""
    # Check for required environment variables
    discord_token = os.getenv('DISCORD_BOT_TOKEN')
    if not discord_token:
        logger.error("DISCORD_BOT_TOKEN environment variable is required")
        return
    
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_key:
        logger.error("ANTHROPIC_API_KEY environment variable is required")
        return
    
    # Create and run bot
    bot = ContextAwareBot()
    
    # Add commands to bot
    bot.add_command(help_command)
    bot.add_command(info_command)
    bot.add_command(context_command)
    
    try:
        bot.run(discord_token)
    except Exception as e:
        logger.error(f"Error running bot: {e}")

if __name__ == "__main__":
    main() 