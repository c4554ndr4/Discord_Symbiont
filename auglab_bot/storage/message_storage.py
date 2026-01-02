"""
Message Storage System for 24-Hour Conversation Context
Stores Discord messages with automatic cleanup and context building
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import discord

logger = logging.getLogger(__name__)

class MessageStorage:
    """Storage system for Discord messages with 24-hour sliding window."""
    
    def __init__(self, db_path: str = "bot_memory.db"):
        self.db_path = db_path
        self.init_message_tables()
        logger.info("✅ MessageStorage initialized")
    
    def init_message_tables(self):
        """Initialize message storage tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Messages table for 24h context
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recent_messages (
                    message_id INTEGER PRIMARY KEY,
                    channel_id INTEGER NOT NULL,
                    guild_id INTEGER,
                    author_id INTEGER NOT NULL,
                    author_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    message_type TEXT DEFAULT 'user',
                    reply_to_id INTEGER,
                    attachments TEXT,
                    created_at_utc TEXT NOT NULL
                )
            ''')
            
            # Index for efficient queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_channel_timestamp 
                ON recent_messages(channel_id, timestamp DESC)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cleanup_timestamp 
                ON recent_messages(timestamp)
            ''')
            
            logger.info("📊 Message storage tables initialized")
    
    def store_message(self, message: discord.Message) -> bool:
        """Store a Discord message for conversation context."""
        try:
            # Extract message data
            message_data = {
                'message_id': message.id,
                'channel_id': message.channel.id,
                'guild_id': getattr(message.guild, 'id', None),
                'author_id': message.author.id,
                'author_name': getattr(message.author, 'display_name', str(message.author)),
                'content': message.content,
                'timestamp': message.created_at.isoformat(),
                'message_type': 'bot' if message.author.bot else 'user',
                'reply_to_id': getattr(getattr(message, 'reference', None), 'message_id', None),
                'attachments': json.dumps([{
                    'filename': att.filename,
                    'url': att.url,
                    'content_type': att.content_type
                } for att in message.attachments]) if getattr(message, 'attachments', None) else None,
                'created_at_utc': datetime.utcnow().isoformat()
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO recent_messages 
                    (message_id, channel_id, guild_id, author_id, author_name, 
                     content, timestamp, message_type, reply_to_id, attachments, created_at_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message_data['message_id'], message_data['channel_id'], 
                    message_data['guild_id'], message_data['author_id'],
                    message_data['author_name'], message_data['content'],
                    message_data['timestamp'], message_data['message_type'],
                    message_data['reply_to_id'], message_data['attachments'],
                    message_data['created_at_utc']
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store message {message.id}: {e}")
            return False
    
    def get_channel_context(self, channel_id: int, hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent messages from a channel for conversation context."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT message_id, author_id, author_name, content, timestamp, 
                           message_type, reply_to_id, attachments
                    FROM recent_messages 
                    WHERE channel_id = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                ''', (channel_id, cutoff_time.isoformat(), limit))
                
                messages = []
                for row in cursor.fetchall():
                    message_dict = dict(row)
                    
                    # Parse attachments if present
                    if message_dict['attachments']:
                        try:
                            message_dict['attachments'] = json.loads(message_dict['attachments'])
                        except:
                            message_dict['attachments'] = []
                    else:
                        message_dict['attachments'] = []
                    
                    messages.append(message_dict)
                
                logger.info(f"📥 Retrieved {len(messages)} messages from channel {channel_id} (last {hours}h)")
                return messages
                
        except Exception as e:
            logger.error(f"❌ Failed to get channel context for {channel_id}: {e}")
            return []
    
    def get_dm_context(self, user_id: int, hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent DM messages with a user for conversation context."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT message_id, author_id, author_name, content, timestamp, 
                           message_type, reply_to_id, attachments
                    FROM recent_messages 
                    WHERE guild_id IS NULL 
                      AND (author_id = ? OR channel_id IN (
                          SELECT DISTINCT channel_id FROM recent_messages 
                          WHERE author_id = ? AND guild_id IS NULL
                      ))
                      AND timestamp >= ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                ''', (user_id, user_id, cutoff_time.isoformat(), limit))
                
                messages = []
                for row in cursor.fetchall():
                    message_dict = dict(row)
                    
                    # Parse attachments if present
                    if message_dict['attachments']:
                        try:
                            message_dict['attachments'] = json.loads(message_dict['attachments'])
                        except:
                            message_dict['attachments'] = []
                    else:
                        message_dict['attachments'] = []
                    
                    messages.append(message_dict)
                
                logger.info(f"💬 Retrieved {len(messages)} DM messages with user {user_id} (last {hours}h)")
                return messages
                
        except Exception as e:
            logger.error(f"❌ Failed to get DM context for user {user_id}: {e}")
            return []
    
    def format_context_for_conversation(self, messages: List[Dict[str, Any]], bot_user_id: int) -> List[Dict[str, str]]:
        """Format stored messages for OpenRouter/LLM conversation context."""
        formatted_messages = []
        
        for msg in messages:
            # Determine role based on author
            if msg['author_id'] == bot_user_id:
                role = 'assistant'
            else:
                role = 'user'
            
            # Build content
            content = msg['content']
            
            # Add attachment info if present
            if msg['attachments']:
                attachment_info = []
                for att in msg['attachments']:
                    attachment_info.append(f"[Attachment: {att['filename']}]")
                if attachment_info:
                    content += " " + " ".join(attachment_info)
            
            # Add reply context if this is a reply
            if msg['reply_to_id']:
                content = f"[Reply] {content}"
            
            # Skip empty messages
            if not content.strip():
                continue
            
            formatted_messages.append({
                'role': role,
                'content': content
            })
        
        logger.info(f"🔄 Formatted {len(formatted_messages)} messages for LLM context")
        return formatted_messages
    
    def cleanup_old_messages(self, hours: int = 48):
        """Remove messages older than specified hours to prevent database bloat."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    DELETE FROM recent_messages 
                    WHERE timestamp < ?
                ''', (cutoff_time.isoformat(),))
                
                deleted_count = cursor.rowcount
                
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old messages (older than {hours}h)")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old messages: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored messages."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total messages
                total = conn.execute('SELECT COUNT(*) FROM recent_messages').fetchone()[0]
                
                # Messages by type
                user_count = conn.execute(
                    "SELECT COUNT(*) FROM recent_messages WHERE message_type = 'user'"
                ).fetchone()[0]
                
                bot_count = conn.execute(
                    "SELECT COUNT(*) FROM recent_messages WHERE message_type = 'bot'"
                ).fetchone()[0]
                
                # Unique channels
                channels = conn.execute(
                    'SELECT COUNT(DISTINCT channel_id) FROM recent_messages'
                ).fetchone()[0]
                
                # Recent activity (last 24h)
                cutoff_24h = (datetime.now() - timedelta(hours=24)).isoformat()
                recent_24h = conn.execute(
                    'SELECT COUNT(*) FROM recent_messages WHERE timestamp >= ?',
                    (cutoff_24h,)
                ).fetchone()[0]
                
                return {
                    'total_messages': total,
                    'user_messages': user_count,
                    'bot_messages': bot_count,
                    'unique_channels': channels,
                    'messages_last_24h': recent_24h
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get storage stats: {e}")
            return {}
    
    async def backfill_channel_history(self, channel: discord.TextChannel, hours: int = 24, limit: int = 100):
        """Backfill recent message history for a channel (use sparingly due to rate limits)."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            stored_count = 0
            
            logger.info(f"📥 Backfilling {hours}h history for #{channel.name} (limit: {limit})")
            
            async for message in channel.history(limit=limit, after=cutoff_time):
                if self.store_message(message):
                    stored_count += 1
            
            logger.info(f"✅ Backfilled {stored_count} messages for #{channel.name}")
            return stored_count
            
        except discord.Forbidden:
            logger.warning(f"⚠️ No permission to read history in #{channel.name}")
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to backfill #{channel.name}: {e}")
            return 0 