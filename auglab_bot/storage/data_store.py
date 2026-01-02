"""
Data storage for Augmentation Lab Bot
"""

import json
import sqlite3
import logging
from typing import Dict, List, Optional
import discord

logger = logging.getLogger(__name__)


class ChannelDataStore:
    """Stores and manages all channel data for later reference."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the SQLite database for data storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS channel_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER NOT NULL,
                    channel_name TEXT NOT NULL,
                    guild_id INTEGER,
                    message_id INTEGER NOT NULL,
                    author_id INTEGER NOT NULL,
                    author_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    attachments TEXT,
                    embeds TEXT
                )
            ''')
            
            # Member cache table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS member_cache (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    guild_id INTEGER NOT NULL,
                    roles TEXT NOT NULL,
                    joined_at TEXT,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # Migration: Add guild_id column if it doesn't exist (for existing databases)
            try:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(member_cache)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'guild_id' not in columns:
                    logger.info("🔄 Migrating member_cache table: adding guild_id column")
                    conn.execute('ALTER TABLE member_cache ADD COLUMN guild_id INTEGER NOT NULL DEFAULT 0')
                    logger.info("✅ Migration complete: guild_id column added")
                    
            except Exception as e:
                logger.warning(f"Migration note: {e} (this might be expected if table is new)")
            
            conn.commit()
        
    def store_member(self, member: discord.Member):
        """Store member information in the cache."""
        try:
            roles = json.dumps([role.name for role in member.roles])
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO member_cache
                    (user_id, username, display_name, guild_id, roles, joined_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    member.id,
                    member.name,
                    member.display_name,
                    member.guild.id,
                    roles,
                    member.joined_at.isoformat() if member.joined_at else None,
                    discord.utils.utcnow().isoformat()
                ))
        except Exception as e:
            logger.error(f"Error storing member: {e}")
    
    def store_message(self, message: discord.Message):
        """Store a message in the database."""
        try:
            attachments = json.dumps([{
                'filename': att.filename,
                'url': att.url,
                'size': att.size
            } for att in message.attachments])
            
            embeds = json.dumps([embed.to_dict() for embed in message.embeds])
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO channel_data
                    (channel_id, channel_name, guild_id, message_id, author_id, 
                     author_name, content, timestamp, attachments, embeds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message.channel.id,
                    getattr(message.channel, 'name', 'DM'),  # Handle DM channels
                    message.guild.id if message.guild else 0,  # Use 0 for DMs instead of None
                    message.id,
                    message.author.id,
                    message.author.display_name,
                    message.content,
                    message.created_at.isoformat(),
                    attachments,
                    embeds
                ))
        except Exception as e:
            logger.error(f"Error storing message: {e}")
    
    def get_channel_history(self, channel_id: int, limit: int = 50) -> List[Dict]:
        """Get stored channel history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT author_name, content, timestamp
                FROM channel_data
                WHERE channel_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (channel_id, limit))
            
            return [{'author': row[0], 'content': row[1], 'timestamp': row[2]} 
                   for row in cursor.fetchall()]
    
    def search_content(self, query: str, channel_id: Optional[int] = None) -> List[Dict]:
        """Search stored content."""
        sql = '''
            SELECT channel_name, author_name, content, timestamp
            FROM channel_data
            WHERE content LIKE ?
        '''
        params = [f'%{query}%']
        
        if channel_id:
            sql += ' AND channel_id = ?'
            params.append(channel_id)
            
        sql += ' ORDER BY timestamp DESC LIMIT 20'
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            return [{'channel': row[0], 'author': row[1], 'content': row[2], 'timestamp': row[3]} 
                   for row in cursor.fetchall()]
    
    def get_member_info(self, user_id: int, guild_id: int = None) -> Optional[Dict]:
        """Get cached member information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                sql = '''
                    SELECT user_id, username, display_name, guild_id, roles, joined_at, last_updated
                    FROM member_cache
                    WHERE user_id = ?
                '''
                params = [user_id]
                
                if guild_id:
                    sql += ' AND guild_id = ?'
                    params.append(guild_id)
                    
                cursor = conn.execute(sql, params)
                row = cursor.fetchone()
                
                if row:
                    return {
                        'user_id': row[0],
                        'username': row[1],
                        'display_name': row[2],
                        'guild_id': row[3],
                        'roles': json.loads(row[4]),
                        'joined_at': row[5],
                        'last_updated': row[6]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting member info: {e}")
            return None 