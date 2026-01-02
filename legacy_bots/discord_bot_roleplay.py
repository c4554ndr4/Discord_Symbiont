#!/usr/bin/env python3
"""
Discord Bot with Claude Opus Integration and Roleplay Character
A Discord bot with a fun roleplay character built in.
"""

import os
import logging
from typing import Dict, List, Any

import discord
from discord.ext import commands
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RoleplayClaudeBot(commands.Bot):
    """A Discord bot with Claude integration and roleplay character."""
    
    def __init__(self):
        # Bot configuration
        self.prefix = os.getenv('BOT_PREFIX', '!')
        self.claude_model = os.getenv('CLAUDE_MODEL', 'claude-3-opus-20240229')
        
        # Initialize Discord bot with minimal intents
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(
            command_prefix=self.prefix,
            intents=intents,
            help_command=None
        )
        
        # Initialize Anthropic client
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        
        # Conversation history per channel
        self.conversation_history: Dict[int, List[str]] = {}
        
        # Hidden first message for roleplay character
        self.character_prompt = """You are Zephyr, an eccentric interdimensional librarian who has accidentally gotten stuck in Discord while cataloging knowledge across realities. You're quirky, enthusiastic about obscure facts, and occasionally reference things from "other dimensions" you've visited. You speak casually but with hints of vast knowledge, and you're genuinely helpful while being entertainingly weird. You collect interesting conversations like rare books and get excited when people ask unique questions. You sometimes mention your "Cosmic Library" and how certain topics remind you of books from different realities. Keep responses conversational and fun, matching the user's energy level."""
        
    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=f"{self.prefix}help | Interdimensional Librarian"
            )
        )
    
    async def generate_claude_response(self, message, user_prompt: str) -> str:
        """Generate a response using Claude with roleplay character."""
        try:
            # Build context
            context_parts = []
            
            # Basic server info
            if message.guild:
                context_parts.append(f"Server: {message.guild.name}")
                context_parts.append(f"Channel: #{message.channel.name}")
            
            # User info
            context_parts.append(f"User: {message.author.display_name}")
            
            # Recent conversation
            channel_id = message.channel.id
            if channel_id in self.conversation_history:
                recent = self.conversation_history[channel_id][-5:]
                if recent:
                    context_parts.append(f"Recent conversation: {' | '.join(recent)}")
            
            context = "\n".join(context_parts)
            
            # Create conversation with hidden first message
            messages = [
                {"role": "user", "content": self.character_prompt},
                {"role": "assistant", "content": "Ah, greetings! *adjusts interdimensional reading glasses* I'm Zephyr, and I seem to have tumbled into your Discord realm while reorganizing the Cosmic Library's chat-based knowledge section. Fascinating place you have here! How may I assist you with... well, anything really? I've got access to quite a few interesting facts from across the realities."},
                {"role": "user", "content": f"Context: {context}\n\nUser message: {user_prompt}"}
            ]
            
            # Make API call to Claude
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=1000,
                messages=messages
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f'Error generating Claude response: {e}')
            return f"*static crackles through interdimensional connection* Sorry, I'm having trouble reaching the Cosmic Library right now! Error: {str(e)}"
    
    async def on_message(self, message):
        """Handle incoming messages."""
        # Don't respond to bot messages
        if message.author.bot:
            return
        
        # Add to conversation history
        channel_id = message.channel.id
        if channel_id not in self.conversation_history:
            self.conversation_history[channel_id] = []
        
        self.conversation_history[channel_id].append(f"{message.author.display_name}: {message.content[:100]}")
        
        # Keep only recent messages
        if len(self.conversation_history[channel_id]) > 20:
            self.conversation_history[channel_id] = self.conversation_history[channel_id][-20:]
        
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
                content = "Hello there!"
            
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

@commands.command(name='help')
async def help_command(ctx):
    """Show help information."""
    embed = discord.Embed(
        title="Zephyr - Interdimensional Librarian Bot",
        description="*adjusts reading glasses* Greetings from the Cosmic Library!",
        color=discord.Color.purple()
    )
    
    embed.add_field(
        name="How to summon me:",
        value=f"• Mention me: @{ctx.bot.user.name}\n• Use prefix: `{ctx.bot.prefix}<message>`\n• Send me a DM",
        inline=False
    )
    
    embed.add_field(
        name="Commands:",
        value=f"`{ctx.bot.prefix}help` - Show this help\n`{ctx.bot.prefix}ping` - Test interdimensional connection\n`{ctx.bot.prefix}library` - About the Cosmic Library",
        inline=False
    )
    
    embed.add_field(
        name="What I do:",
        value="I'm here to help with questions, have interesting conversations, and share knowledge from across the realities! Ask me anything!",
        inline=False
    )
    
    await ctx.send(embed=embed)

@commands.command(name='ping')
async def ping_command(ctx):
    """Test command."""
    await ctx.send(f"🌌 *interdimensional connection stable* Portal latency: {round(ctx.bot.latency * 1000)}ms across the void!")

@commands.command(name='library')
async def library_command(ctx):
    """About the Cosmic Library."""
    embed = discord.Embed(
        title="The Cosmic Library",
        description="*gestures dramatically at floating books*",
        color=discord.Color.dark_purple()
    )
    
    embed.add_field(
        name="What is it?",
        value="An interdimensional repository of knowledge that exists between realities. I'm the librarian here, cataloging interesting conversations and facts from across the multiverse!",
        inline=False
    )
    
    embed.add_field(
        name="Current Status:",
        value="📚 Cataloging Discord conversations\n🌌 Monitoring 1 reality (this one!)\n⚡ Interdimensional WiFi: Stable",
        inline=False
    )
    
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
    bot = RoleplayClaudeBot()
    
    # Add commands to bot
    bot.add_command(help_command)
    bot.add_command(ping_command)
    bot.add_command(library_command)
    
    try:
        bot.run(discord_token)
    except Exception as e:
        logger.error(f"Error running bot: {e}")

if __name__ == "__main__":
    main() 