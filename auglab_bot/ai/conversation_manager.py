"""
Conversation Management for Augmentation Lab Bot
Handles conversation threading and context management
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import discord

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation context with 24-hour message history for AI responses."""
    
    def __init__(self, message_storage=None, recent_message_limit: int = 50, bot_user_id: int = None):
        # Use message storage for 24h context instead of simple cache
        self.message_storage = message_storage
        self.recent_message_limit = recent_message_limit
        self.bot_user_id = bot_user_id
        
        # Fallback to old system if no message storage provided
        self.recent_contexts = {}  # key -> list of recent messages (fallback only)
        
    def get_conversation_key(self, message) -> str:
        """Generate a unique key for conversation context."""
        if isinstance(message.channel, discord.DMChannel):
            return f"dm_{message.author.id}"
        else:
            return f"channel_{message.channel.id}"
    
    def get_recent_context(self, message) -> List[Dict[str, Any]]:
        """Get recent conversation context for a message from 24h message history."""
        if self.message_storage:
            # Use 24h message storage for rich context
            if isinstance(message.channel, discord.DMChannel):
                # For DMs, get DM history with the user
                messages = self.message_storage.get_dm_context(
                    user_id=message.author.id, 
                    hours=24, 
                    limit=self.recent_message_limit
                )
            else:
                # For channels, get channel history
                messages = self.message_storage.get_channel_context(
                    channel_id=message.channel.id, 
                    hours=24, 
                    limit=self.recent_message_limit
                )
            
            # Convert to old format for compatibility
            return [{
                'role': 'assistant' if msg['message_type'] == 'bot' else 'user',
                'content': msg['content'],
                'timestamp': datetime.fromisoformat(msg['timestamp']),
                'user_id': msg['author_id'],
                'channel_id': message.channel.id
            } for msg in messages]
        else:
            # Fallback to old in-memory system
            key = self.get_conversation_key(message)
            return self.recent_contexts.get(key, [])
    
    def add_message_to_context(self, message, role: str, content: str):
        """Add message to recent context (simple FIFO queue)."""
        key = self.get_conversation_key(message)
        
        if key not in self.recent_contexts:
            self.recent_contexts[key] = []
        
        # Add new message
        context_entry = {
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'user_id': message.author.id if hasattr(message, 'author') else None,
            'channel_id': message.channel.id if hasattr(message, 'channel') else None
        }
        
        self.recent_contexts[key].append(context_entry)
        
        # Keep only recent messages (simple FIFO)
        if len(self.recent_contexts[key]) > self.recent_message_limit:
            self.recent_contexts[key] = self.recent_contexts[key][-self.recent_message_limit:]
        
        logger.info(f"💬 Recent context for {getattr(message.author, 'display_name', 'Unknown')}: {len(self.recent_contexts[key])} messages")
    
    def build_conversation_messages(self, message, content: str, use_expensive: bool = False) -> List[Dict[str, str]]:
        """Build conversation messages array for API call using 24h message history."""
        if self.message_storage:
            # Use MessageStorage for rich 24h context
            # Try to get bot_user_id from guild context, fallback to stored bot_user_id
            # Safely get bot_user_id with fallback chain
            bot_user_id = self.bot_user_id
            if message.guild and message.guild.me:
                bot_user_id = message.guild.me.id
            elif hasattr(message, 'guild') and message.guild and hasattr(message.guild, 'get_member') and self.bot_user_id:
                # Try to get bot member from guild cache
                bot_member = message.guild.get_member(self.bot_user_id)
                if bot_member:
                    bot_user_id = bot_member.id
            
            if isinstance(message.channel, discord.DMChannel):
                # For DMs, get DM history
                stored_messages = self.message_storage.get_dm_context(
                    user_id=message.author.id, 
                    hours=24, 
                    limit=self.recent_message_limit
                )
            else:
                # For channels, get channel history  
                stored_messages = self.message_storage.get_channel_context(
                    channel_id=message.channel.id, 
                    hours=24, 
                    limit=self.recent_message_limit
                )
            
            # Format for conversation
            messages = self.message_storage.format_context_for_conversation(
                stored_messages, bot_user_id
            )
            
            # Add current user message (avoid duplicates)
            if not messages or messages[-1]['content'] != content:
                messages.append({
                    'role': 'user', 
                    'content': content
                })
            
            return messages
        else:
            # Fallback to old system
            recent_context = self.get_recent_context(message)
            
            messages = []
            
            # Add recent context (but not the current message we're about to add)
            for context_msg in recent_context:
                # Skip if this is the same content we're about to add
                if context_msg['content'] == content and context_msg['role'] == 'user':
                    continue
                    
                messages.append({
                    'role': context_msg['role'],
                    'content': context_msg['content']
                })
            
            # Add current user message
            messages.append({
                'role': 'user', 
                'content': content
            })
            
            return messages
    
    def build_system_message(self, message, config_manager, working_memory_manager, memory_system) -> str:
        """Build system message with context - memory system handles historical context."""
        # Get base system prompt
        main_prompt = config_manager.get('system_prompt.main_prompt', '')
        
        # Handle both string and array formats for backward compatibility
        if isinstance(main_prompt, list):
            main_prompt = '\n'.join(main_prompt)
        
        # Get working memory (dynamic context and current intentions)
        working_memory = working_memory_manager.get_current_working_memory()
        
        # Get relevant memories (memory system handles this automatically)
        query_text = message.content if hasattr(message, 'content') else ""
        user_id = message.author.id if hasattr(message, 'author') and hasattr(message.author, 'id') else None
        memories = memory_system.query_memories(query_text, user_id=user_id, limit=3)
        
        logger.info(f"🔍 MEMORY QUERY: '{query_text[:100]}...' | user_id={user_id} | limit=3")
        
        if memories:
            logger.info(f"📚 RETRIEVED {len(memories)} MEMORIES:")
            for i, memory in enumerate(memories, 1):
                similarity = memory.get('similarity', 0.0)
                content_preview = memory['content'][:200]
                logger.info(f"  {i}. [{memory.get('memory_type', 'unknown')}] (sim: {similarity:.3f}) {memory.get('created_at', 'unknown')}: {content_preview}...")
        else:
            logger.info("📚 NO MEMORIES RETRIEVED")
        
        context_parts = [main_prompt]
        
        # Add working memory if available (current dynamic context/intentions)
        if working_memory:
            context_parts.append(f"## Current Working Memory\n{working_memory}")
        
        # Add relevant memories (this handles historical context automatically)
        if memories:
            logger.info(f"🧠 ADDING {len(memories)} MEMORIES TO SYSTEM MESSAGE:")
            context_parts.append("## Relevant Context from Memory")
            for i, memory in enumerate(memories, 1):
                similarity = memory.get('similarity', 0.0)
                content_preview = memory['content'][:200]
                context_parts.append(f"Memory (similarity: {similarity:.2f}): {content_preview}")
                logger.info(f"  {i}. Added memory (sim: {similarity:.3f}): {content_preview[:100]}...")
        else:
            logger.info("🧠 NO MEMORIES ADDED TO SYSTEM MESSAGE")
        
        # Add user context
        if hasattr(message, 'author'):
            user_info = f"Current user: {getattr(message.author, 'display_name', 'Unknown')} (@{getattr(message.author, 'name', 'unknown')})"
            context_parts.append(f"## User Context\n{user_info}")
        
        # Add channel context
        if hasattr(message, 'channel'):
            if isinstance(message.channel, discord.DMChannel):
                channel_info = "This is a direct message conversation."
            else:
                channel_name = getattr(message.channel, 'name', 'unknown-channel')
                channel_info = f"Current channel: #{channel_name}"
            context_parts.append(f"## Channel Context\n{channel_info}")
            
        # Add recent conversation context (just the last few messages in this channel/DM)
        recent_context = self.get_recent_context(message)
        if recent_context:
            context_parts.append("## Recent Conversation")
            for ctx_msg in recent_context:
                role = ctx_msg['role']
                content = ctx_msg['content'][:150] + "..." if len(ctx_msg['content']) > 150 else ctx_msg['content']
                context_parts.append(f"{role}: {content}")
        
        # Add function calling instructions
        context_parts.append("""
## Available Functions
You can call functions to perform actions using XML-style tags. Here are your available functions:

**Messaging Functions:**
- `<message_user users="[user1,user2]" destination="channel_name">message content</message_user>` - Send messages to users in channels
- `<find_member_by_name name="username">search for user</find_member_by_name>` - Find Discord members by name

**Memory Functions:**
- `<query_memory>search terms</query_memory>` - Search your memory for relevant information
- `<store_observation user_id="123" channel_id="456" tags="tag1,tag2">observation to store</store_observation>` - Store important observations

**Utility Functions:**
- `<think>internal reasoning</think>` - Internal thinking (not shown to users)
- `<get_budget>check spending</get_budget>` - Check current budget status
- `<get_current_time>get time</get_current_time>` - Get current date/time

**Examples:**
- To ping a user: `<message_user users="[vie]" destination="🌎-res-general">Hey Vie! Cassandra wants to connect with you.</message_user>`
- To search memory: `<query_memory>summit planning discussions</query_memory>`
- To find a user: `<find_member_by_name name="vie">looking for vie</find_member_by_name>`

Always use these functions when users ask you to take actions like messaging, pinging, or searching.
""")

        # Add short response mode instructions if enabled
        short_response_mode = config_manager.get('bot.short_response_mode', False)
        if short_response_mode:
            short_limit = config_manager.get('bot.short_response_limit', 240)
            context_parts.append(f"""
## CRITICAL: SHORT RESPONSE MODE ENABLED
You MUST keep your response to {short_limit} characters or less (including any user tags).
Be concise, direct, and impactful. Remove unnecessary words while maintaining your personality.
This is a hard limit - responses over {short_limit} characters will be rejected and you'll need to rewrite them.
""")
        
        return '\n\n'.join(context_parts)
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation contexts."""
        total_contexts = len(self.recent_contexts)
        total_messages = sum(len(context) for context in self.recent_contexts.values())
        
        return {
            'total_contexts': total_contexts,
            'total_recent_messages': total_messages,
            'average_messages_per_context': total_messages / total_contexts if total_contexts > 0 else 0,
            'recent_message_limit': self.recent_message_limit
        }
    
    def cleanup_empty_contexts(self):
        """Remove empty contexts (simple cleanup)."""
        empty_keys = [key for key, context in self.recent_contexts.items() if not context]
        for key in empty_keys:
            del self.recent_contexts[key]
        
        if empty_keys:
            logger.info(f"Cleaned up {len(empty_keys)} empty conversation contexts")
    
    def clear_context(self, key: str):
        """Clear a specific conversation context."""
        if key in self.recent_contexts:
            del self.recent_contexts[key]
    
    def clear_all_contexts(self):
        """Clear all conversation contexts."""
        self.recent_contexts.clear()

    # Deprecated methods for backward compatibility
    def get_conversation_thread(self, message) -> List[Dict[str, Any]]:
        """Deprecated: use get_recent_context instead."""
        return self.get_recent_context(message)
    
    def add_to_conversation_thread(self, message, role: str, content: str):
        """Deprecated: use add_message_to_context instead."""
        return self.add_message_to_context(message, role, content) 