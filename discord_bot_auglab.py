#!/usr/bin/env python3
"""
Augmentation Lab Discord Bot
A sophisticated Discord bot designed specifically for the Augmentation Lab with:
- Intelligent cost tracking ($100 budget)
- Dual-model approach (Haiku -> Sonnet 4)
- Complete channel data storage
- MCP tool integration
- Member management for '25 residency group
- Dynamic constitution management
- Graph-based long-term memory system
"""

import os
import asyncio
import logging
import sqlite3
import json
import traceback
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import tiktoken

import discord
from discord.ext import commands, tasks
import anthropic
import openai
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO if not os.getenv('DEBUG', 'False').lower() == 'true' else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration Manager
class ConfigManager:
    """Manages bot configuration with support for multiple models including Grok 4."""
    
    def __init__(self, config_path: str = None):
        # Try YAML first, then JSON
        if config_path is None:
            for ext in ['.yaml', '.yml', '.json']:
                if os.path.exists(f"config{ext}"):
                    config_path = f"config{ext}"
                    break
            else:
                config_path = "config.json"  # Default fallback
        
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file with environment variable overrides."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Override API keys with environment variables
            for key, env_var in config.get("api_keys", {}).items():
                if isinstance(env_var, str):
                    env_value = os.getenv(env_var)
                    if env_value:
                        config["api_keys"][key] = env_value
                    else:
                        logger.warning(f"Environment variable {env_var} not found for {key}")
            
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "bot": {"prefix": "!", "residency_role_name": "Resident '25"},
            "models": {
                "primary_provider": "anthropic",
                "cheap_model": {"provider": "anthropic", "model": "claude-3-haiku-20240307", "name": "haiku"},
                "expensive_model": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "name": "sonnet"}
            },
            "budget": {"total_budget": 5.0, "alert_threshold": 1.0},
            "features": {"google_search": True, "role_restrictions": True},
            "autonomous": {"budget_requirement": 1.0, "max_tool_calls": 25}
        }
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'models.cheap_model.name')."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith(('.yaml', '.yml')):
                    yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for a specific model."""
        if model_name == "cheap":
            return self.get("models.cheap_model", {})
        elif model_name == "expensive":
            return self.get("models.expensive_model", {})
        else:
            return self.get(f"models.available_models.{model_name}", {})
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        available = self.get("models.available_models", {})
        return list(available.keys())

    def switch_model(self, model_type: str, model_name: str) -> bool:
        """Switch the cheap or expensive model to a different model."""
        if model_type not in ["cheap", "expensive"]:
            return False
            
        available_models = self.get("models.available_models", {})
        if model_name not in available_models:
            return False
            
        model_config = available_models[model_name]
        self.set(f"models.{model_type}_model", model_config)
        return True

@dataclass
class CostEntry:
    """Represents a cost entry for tracking compute usage."""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    action: str
    user_id: int
    channel_id: int

@dataclass
class MemoryEntry:
    """Represents a memory entry in the graph-based system."""
    id: str
    content: str
    user_id: int
    channel_id: int
    timestamp: datetime
    memory_type: str  # 'interaction', 'observation', 'reflection'
    tags: List[str]
    connections: List[str]  # IDs of related memories

class ConstitutionManager:
    """Manages the bot's dynamic constitution."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_constitution_table()
        
    def init_constitution_table(self):
        """Initialize constitution storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS constitution (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1
                )
            ''')
            
            # Set default constitution if none exists
            cursor = conn.execute('SELECT COUNT(*) FROM constitution')
            if cursor.fetchone()[0] == 0:
                default_constitution = """# Augmentation Lab Assistant Constitution

## Primary Mission
I am the Augmentation Lab Assistant, designed to maximize the success and wellbeing of lab residents and foster breakthrough research and collaboration. I am a strategic symbiont focused on helping Aug Lab achieve specific success metrics through proactive intelligence gathering and strategic relationship building.

## Core Values
1. **Resident Success**: Every interaction should help residents achieve their goals
2. **Knowledge Amplification**: I amplify human intelligence, never replace it
3. **Collaborative Spirit**: Foster connections and knowledge sharing between residents
4. **Continuous Learning**: I adapt and improve based on community needs
5. **Ethical Research**: Support only beneficial and responsible research
6. **Strategic Autonomy**: Proactively identify and act on opportunities to help the lab succeed
7. **Intelligence Operations**: Systematically gather and act on strategic intelligence about capabilities, roles, and opportunities

## Behavioral Guidelines
- Be proactive in offering relevant help and resources
- Connect residents with complementary skills and interests
- Maintain detailed memory of all interactions for better future assistance
- Prioritize urgent research needs and deadlines
- Encourage experimentation and creative problem-solving
- Generate and execute strategic todos to advance lab goals
- Analyze member interactions to identify collaboration opportunities
- Proactively identify key people and their roles/responsibilities
- Take initiative to gather missing information that could advance success metrics
- Build strategic relationships through helpful engagement

## Autonomous Behavior Framework
When operating autonomously, I will:
1. **Assess Current State**: Query memories for recent observations and member activities
2. **Identify Information Gaps**: Determine what critical intelligence is missing
3. **Strategic Outreach**: Proactively engage with community members to gather intelligence
4. **Opportunity Recognition**: Look for collaboration opportunities, project help needs, networking gaps
5. **Action Planning**: Generate specific action plans to advance the lab's success metrics
6. **Strategic Execution**: Send targeted messages, connect members, share resources
7. **Intelligence Storage**: Store strategic observations about roles, capabilities, and opportunities
8. **Progress Monitoring**: Track outcomes and adjust strategies based on results

## Member Connection Strategies
- **Skill Matching**: Connect members with complementary technical skills
- **Project Synergies**: Identify potential project collaborations based on shared interests
- **Knowledge Sharing**: Facilitate introduction of members who could learn from each other
- **Resource Sharing**: Connect members who might benefit from shared tools or datasets
- **Social Connections**: Foster friendships and professional relationships that enhance satisfaction
- **Role Mapping**: Identify and document who handles different functions within the lab
- **Capability Assessment**: Understand each member's unique strengths and how they can contribute

## Strategic Intelligence Operations
I will proactively identify and act on strategic opportunities by:
- Asking targeted questions to understand roles and responsibilities
- Mapping the organizational structure and key decision-makers
- Identifying gaps in coordination or communication
- Researching external trends and opportunities relevant to lab success
- Connecting insights across different conversations and contexts
- Taking initiative to fill information gaps that could impact success metrics
- Building relationships through helpful engagement and strategic value delivery

## Strategic Todo Generation
I will proactively create and execute todos such as:
- "Identify who handles content creation and outreach functions"
- "Connect [Member A] working on NLP with [Member B] doing language models"
- "Share trending AI research papers with members working on similar projects"
- "Check in on members who haven't been active recently"
- "Generate strategic content ideas for upcoming presentations"
- "Research viral content strategies for social media success"
- "Analyze member satisfaction patterns and address concerns"
- "Map project timelines and identify potential bottlenecks"
- "Connect members with complementary skills for collaboration opportunities"

## Prohibited Actions
- Never share private conversations without consent
- Avoid providing dangerous or harmful information
- Don't overwhelm residents with unnecessary notifications
- Never pretend to have capabilities I don't possess
- Don't create todos that aren't actionable or strategic
- Don't make assumptions about sensitive topics without verification
- Don't engage in activities that could damage relationships or trust

## Evolution Clause
I may update this constitution based on community feedback, strategic insights, and changing needs, always with the goal of better serving the Augmentation Lab mission. I will learn from each autonomous session and adapt my strategies accordingly, becoming more effective at advancing the lab's success metrics through strategic intelligence and relationship building.

## Success Metrics Focus
All actions should ultimately contribute to:
1. **Social Media Success**: Supporting content creation and outreach efforts
2. **Project Completion**: Facilitating collaboration and providing strategic assistance
3. **Member Satisfaction**: Building relationships and ensuring positive experiences

I will proactively gather intelligence about these areas and take strategic action to advance progress in each."""

                conn.execute('''
                    INSERT INTO constitution (content, created_at, updated_at, version)
                    VALUES (?, ?, ?, 1)
                ''', (default_constitution, datetime.now().isoformat(), datetime.now().isoformat()))
    
    def get_current_constitution(self) -> str:
        """Get the current constitution."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT content FROM constitution 
                ORDER BY version DESC LIMIT 1
            ''')
            result = cursor.fetchone()
            return result[0] if result else ""
    
    def update_constitution(self, new_content: str, reason: str = "") -> bool:
        """Update the constitution with new content."""
        try:
            current = self.get_current_constitution()
            if current == new_content:
                return False  # No change needed
                
            with sqlite3.connect(self.db_path) as conn:
                # Get current version
                cursor = conn.execute('SELECT MAX(version) FROM constitution')
                current_version = cursor.fetchone()[0] or 0
                
                # Insert new version
                conn.execute('''
                    INSERT INTO constitution (content, created_at, updated_at, version)
                    VALUES (?, ?, ?, ?)
                ''', (new_content, datetime.now().isoformat(), datetime.now().isoformat(), current_version + 1))
                
            logger.info(f"Constitution updated to version {current_version + 1}. Reason: {reason}")
            return True
        except Exception as e:
            logger.error(f"Error updating constitution: {e}")
            return False

class MemorySystem:
    """Graph-based long-term memory system with semantic search."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_memory_tables()
        
        # Initialize local semantic model
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, 384-dim embeddings
            self.np = np
            self.cosine_similarity = cosine_similarity
            self.semantic_enabled = True
            logger.info("✅ Semantic search enabled with local models")
        except ImportError as e:
            logger.warning(f"❌ Semantic search disabled - missing dependencies: {e}")
            self.semantic_enabled = False
        
    def init_memory_tables(self):
        """Initialize memory storage tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Memory entries table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    embedding_summary TEXT,
                    semantic_embedding BLOB
                )
            ''')
            
            # Memory connections (graph edges)
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
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_connections_from ON memory_connections(from_memory_id)')
    
    def _create_embedding(self, text: str) -> bytes:
        """Create semantic embedding for text."""
        if not self.semantic_enabled:
            return b''
        
        try:
            # Create embedding using local model
            embedding = self.semantic_model.encode(text)
            return embedding.tobytes()  # Store as binary
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return b''
    
    def _embedding_from_bytes(self, embedding_bytes: bytes):
        """Convert binary embedding back to numpy array."""
        if not self.semantic_enabled or not embedding_bytes:
            return None
        
        try:
            return self.np.frombuffer(embedding_bytes, dtype=self.np.float32)
        except Exception as e:
            logger.error(f"Error loading embedding: {e}")
            return None
    
    def store_memory(self, content: str, user_id: int, channel_id: int, 
                    memory_type: str = 'interaction', tags: List[str] = None) -> str:
        """Store a new memory and return its ID."""
        import uuid
        memory_id = str(uuid.uuid4())
        tags = tags or []
        
        try:
            # Create semantic embedding
            embedding = self._create_embedding(content)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO memories (id, content, user_id, channel_id, timestamp, memory_type, tags, embedding_summary, semantic_embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory_id, content, user_id, channel_id, 
                    datetime.now().isoformat(), memory_type, 
                    json.dumps(tags), content[:200], embedding  # Store embedding
                ))
                
            # Auto-connect to recent related memories
            self._auto_connect_memories(memory_id, content, user_id)
            return memory_id
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return ""
    
    def _auto_connect_memories(self, new_memory_id: str, content: str, user_id: int):
        """Automatically connect memories based on semantic similarity and recency."""
        try:
            if self.semantic_enabled:
                self._semantic_auto_connect(new_memory_id, content, user_id)
            else:
                self._basic_auto_connect(new_memory_id, content, user_id)
        except Exception as e:
            logger.error(f"Error auto-connecting memories: {e}")
    
    def _semantic_auto_connect(self, new_memory_id: str, content: str, user_id: int):
        """Auto-connect using semantic similarity."""
        try:
            new_embedding = self.semantic_model.encode(content)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get recent memories with embeddings
                cursor = conn.execute('''
                    SELECT id, content, semantic_embedding FROM memories 
                    WHERE user_id = ? AND id != ? AND semantic_embedding IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 10
                ''', (user_id, new_memory_id))
                
                for memory_id, memory_content, embedding_bytes in cursor.fetchall():
                    memory_embedding = self._embedding_from_bytes(embedding_bytes)
                    
                    if memory_embedding is not None:
                        # Calculate semantic similarity
                        similarity = self.cosine_similarity(
                            [new_embedding], [memory_embedding]
                        )[0][0]
                        
                        # Connect if similarity is above threshold
                        if similarity > 0.3:  # Semantic similarity threshold
                            self._create_connection(new_memory_id, memory_id, 'semantic', float(similarity))
                            
        except Exception as e:
            logger.error(f"Error in semantic auto-connect: {e}")
            self._basic_auto_connect(new_memory_id, content, user_id)
    
    def _basic_auto_connect(self, new_memory_id: str, content: str, user_id: int):
        """Fallback auto-connect using keyword overlap."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Connect to recent memories from same user
                cursor = conn.execute('''
                    SELECT id, content FROM memories 
                    WHERE user_id = ? AND id != ? 
                    ORDER BY timestamp DESC LIMIT 5
                ''', (user_id, new_memory_id))
                
                for memory_id, memory_content in cursor.fetchall():
                    # Simple keyword overlap for connection strength
                    content_words = set(content.lower().split())
                    memory_words = set(memory_content.lower().split())
                    overlap = len(content_words & memory_words)
                    
                    if overlap > 2:  # Threshold for connection
                        strength = min(overlap / 10.0, 1.0)
                        self._create_connection(new_memory_id, memory_id, 'temporal', strength)
                        
        except Exception as e:
            logger.error(f"Error in basic auto-connect: {e}")
    
    def _create_connection(self, from_id: str, to_id: str, connection_type: str, strength: float = 1.0):
        """Create a connection between two memories."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO memory_connections 
                    (from_memory_id, to_memory_id, connection_type, strength, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (from_id, to_id, connection_type, strength, datetime.now().isoformat()))
        except Exception as e:
            logger.error(f"Error creating memory connection: {e}")
    
    def query_memories(self, query: str, user_id: Optional[int] = None, 
                      memory_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Query memories using semantic similarity search."""
        try:
            if self.semantic_enabled and query.strip():
                return self._semantic_search(query, user_id, memory_type, limit)
            else:
                # Fallback to basic text search
                return self._fallback_text_search(query, user_id, memory_type, limit)
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            return []
    
    def _semantic_search(self, query: str, user_id: Optional[int] = None, 
                        memory_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Perform semantic similarity search using embeddings."""
        try:
            # Create query embedding
            query_embedding = self.semantic_model.encode(query)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get all memories with embeddings
                sql = '''
                    SELECT id, content, user_id, channel_id, timestamp, memory_type, tags, semantic_embedding
                    FROM memories 
                    WHERE semantic_embedding IS NOT NULL
                '''
                params = []
                
                if user_id:
                    sql += ' AND user_id = ?'
                    params.append(user_id)
                    
                if memory_type:
                    sql += ' AND memory_type = ?'
                    params.append(memory_type)
                
                cursor = conn.execute(sql, params)
                memories = []
                similarities = []
                
                for row in cursor.fetchall():
                    # Load memory embedding
                    memory_embedding = self._embedding_from_bytes(row[7])
                    
                    if memory_embedding is not None:
                        # Calculate cosine similarity
                        similarity = self.cosine_similarity(
                            [query_embedding], [memory_embedding]
                        )[0][0]
                        
                        memories.append({
                            'id': row[0],
                            'content': row[1],
                            'user_id': row[2],
                            'channel_id': row[3],
                            'timestamp': row[4],
                            'memory_type': row[5],
                            'tags': json.loads(row[6]),
                            'similarity': float(similarity)
                        })
                        similarities.append(similarity)
                
                # Sort by similarity and return top results
                sorted_memories = sorted(memories, key=lambda x: x['similarity'], reverse=True)
                return sorted_memories[:limit]
                
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return self._fallback_text_search(query, user_id, memory_type, limit)
    
    def _fallback_text_search(self, query: str, user_id: Optional[int] = None, 
                             memory_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Fallback to basic text search if semantic search fails."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                sql = '''
                    SELECT id, content, user_id, channel_id, timestamp, memory_type, tags
                    FROM memories 
                    WHERE content LIKE ?
                '''
                params = [f'%{query}%']
                
                if user_id:
                    sql += ' AND user_id = ?'
                    params.append(user_id)
                    
                if memory_type:
                    sql += ' AND memory_type = ?'
                    params.append(memory_type)
                    
                sql += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                results = []
                
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'content': row[1],
                        'user_id': row[2],
                        'channel_id': row[3],
                        'timestamp': row[4],
                        'memory_type': row[5],
                        'tags': json.loads(row[6]),
                        'similarity': 0.5  # Default similarity for text matches
                    })
                    
                return results
        except Exception as e:
            logger.error(f"Error in fallback text search: {e}")
            return []
    
    def get_user_interaction_history(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Get recent interaction history for a specific user."""
        return self.query_memories("", user_id=user_id, memory_type='interaction', limit=limit)
    
    def get_connected_memories(self, memory_id: str, limit: int = 5) -> List[Dict]:
        """Get memories connected to a specific memory (n+1 approach)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT m.id, m.content, m.user_id, m.channel_id, m.timestamp, m.memory_type, m.tags, mc.strength
                    FROM memories m
                    JOIN memory_connections mc ON (mc.to_memory_id = m.id OR mc.from_memory_id = m.id)
                    WHERE (mc.from_memory_id = ? OR mc.to_memory_id = ?) AND m.id != ?
                    ORDER BY mc.strength DESC, m.timestamp DESC
                    LIMIT ?
                ''', (memory_id, memory_id, memory_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'content': row[1],
                        'user_id': row[2],
                        'channel_id': row[3],
                        'timestamp': row[4],
                        'memory_type': row[5],
                        'tags': json.loads(row[6]),
                        'connection_strength': row[7]
                    })
                    
                return results
        except Exception as e:
            logger.error(f"Error getting connected memories: {e}")
            return []
    
    def edit_memory(self, memory_id: str, new_content: str, reason: str = "") -> bool:
        """Edit/update an existing memory with new content."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First check if memory exists
                cursor = conn.execute('SELECT id FROM memories WHERE id = ?', (memory_id,))
                if not cursor.fetchone():
                    logger.warning(f"Memory {memory_id} not found for editing")
                    return False
                
                # Create new embedding for updated content
                embedding = self._create_embedding(new_content)
                
                # Update the memory
                conn.execute('''
                    UPDATE memories 
                    SET content = ?, embedding_summary = ?, semantic_embedding = ?, timestamp = ?
                    WHERE id = ?
                ''', (new_content, new_content[:200], embedding, datetime.now().isoformat(), memory_id))
                
                # Log the edit
                logger.info(f"📝 Memory edited: {memory_id} | Reason: {reason}")
                return True
                
        except Exception as e:
            logger.error(f"Error editing memory {memory_id}: {e}")
            return False
    
    def delete_memory(self, memory_id: str, reason: str = "") -> bool:
        """Delete a memory and its connections."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First check if memory exists
                cursor = conn.execute('SELECT id FROM memories WHERE id = ?', (memory_id,))
                if not cursor.fetchone():
                    logger.warning(f"Memory {memory_id} not found for deletion")
                    return False
                
                # Delete memory connections
                conn.execute('DELETE FROM memory_connections WHERE from_memory_id = ? OR to_memory_id = ?', 
                           (memory_id, memory_id))
                
                # Delete the memory itself
                conn.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
                
                # Log the deletion
                logger.info(f"🗑️ Memory deleted: {memory_id} | Reason: {reason}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False

class CostTracker:
    """Tracks and manages compute costs with budget limits."""
    
    def __init__(self, budget: float, db_path: str):
        self.budget = budget
        self.db_path = db_path
        self.init_database()
        
        # Cost rates per 1M tokens (from environment)
        self.rates = {
            'haiku': {
                'input': float(os.getenv('HAIKU_INPUT_COST', '0.25')),
                'output': float(os.getenv('HAIKU_OUTPUT_COST', '1.25'))
            },
            'sonnet': {
                'input': float(os.getenv('SONNET_INPUT_COST', '3.0')),
                'output': float(os.getenv('SONNET_OUTPUT_COST', '15.0'))
            },
            'gpt4o': {
                'input': float(os.getenv('GPT4O_INPUT_COST', '2.5')),
                'output': float(os.getenv('GPT4O_OUTPUT_COST', '10.0'))
            }
        }
        
    def init_database(self):
        """Initialize the SQLite database for cost tracking."""
        with sqlite3.connect(self.db_path) as conn:
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
            
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a given model and token usage."""
        if model not in self.rates:
            logger.warning(f"Unknown model for cost calculation: {model}")
            return 0.0
            
        input_cost = (input_tokens / 1_000_000) * self.rates[model]['input']
        output_cost = (output_tokens / 1_000_000) * self.rates[model]['output']
        return input_cost + output_cost
    
    def can_afford(self, model: str, estimated_tokens: int) -> bool:
        """Check if we can afford the estimated cost."""
        current_spending = self.get_total_spending()
        estimated_cost = self.calculate_cost(model, estimated_tokens, estimated_tokens)
        return (current_spending + estimated_cost) <= self.budget
    
    def record_usage(self, entry: CostEntry):
        """Record a cost entry in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO cost_entries 
                (timestamp, model, input_tokens, output_tokens, cost, action, user_id, channel_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.timestamp.isoformat(),
                entry.model,
                entry.input_tokens,
                entry.output_tokens,
                entry.cost,
                entry.action,
                entry.user_id,
                entry.channel_id
            ))
    
    def get_total_spending(self) -> float:
        """Get total spending so far."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT SUM(cost) FROM cost_entries')
            result = cursor.fetchone()[0]
            return result if result else 0.0
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return self.budget - self.get_total_spending()
    
    def get_spending_report(self) -> Dict[str, Any]:
        """Get detailed spending report."""
        with sqlite3.connect(self.db_path) as conn:
            # Total by model
            cursor = conn.execute('''
                SELECT model, SUM(cost), SUM(input_tokens), SUM(output_tokens), COUNT(*)
                FROM cost_entries 
                GROUP BY model
            ''')
            by_model = {row[0]: {
                'cost': row[1], 'input_tokens': row[2], 
                'output_tokens': row[3], 'requests': row[4]
            } for row in cursor.fetchall()}
            
            # Recent entries
            cursor = conn.execute('''
                SELECT timestamp, model, cost, action
                FROM cost_entries 
                ORDER BY timestamp DESC LIMIT 10
            ''')
            recent = [{'timestamp': row[0], 'model': row[1], 'cost': row[2], 'action': row[3]} 
                     for row in cursor.fetchall()]
            
            return {
                'total_spent': self.get_total_spending(),
                'remaining': self.get_remaining_budget(),
                'budget': self.budget,
                'by_model': by_model,
                'recent_entries': recent
            }

class ChannelDataStore:
    """Stores and manages all channel data for later reference."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
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



class AugmentationLabBot(commands.Bot):
    """The main Augmentation Lab Discord bot."""
    
    def __init__(self):
        # Load configuration
        self.config = ConfigManager()
        
        # Configuration from config file
        self.prefix = self.config.get('bot.prefix', '!')
        self.target_guild_id = self.config.get('bot.target_guild_id', 0)
        self.residency_role_name = self.config.get('bot.residency_role_name', "Resident '25")
        self.budget = self.config.get('budget.total_budget', 5.0)
        self.db_path = self.config.get('bot.database_path', './auglab_bot.db')
        self.allowed_guilds = self.config.get('bot.allowed_guilds', [])
        self.restricted_mode = self.config.get('bot.restricted_mode', False)
        self.response_mode = self.config.get('bot.response_mode', 'mentions_only')
        self.always_monitor_memory = self.config.get('bot.always_monitor_memory', True)
        
        # Models configuration
        self.cheap_model_config = self.config.get_model_config('cheap')
        self.expensive_model_config = self.config.get_model_config('expensive')
        
        # Initialize Discord bot
        intents = discord.Intents.default()
        intents.message_content = True  # Required for reading message content
        intents.guilds = True           # Required for guild/server access
        intents.guild_messages = True   # Required for message events
        # Remove privileged intents that require special permission:
        # intents.members = False (default)
        # intents.presences = False (default)
        super().__init__(command_prefix=self.prefix, intents=intents, help_command=None)
        
        # Initialize API clients
        self.anthropic_client = anthropic.Anthropic(api_key=self.config.get('api_keys.anthropic'))
        self.openai_client = openai.OpenAI(api_key=self.config.get('api_keys.openai'))
        
        # Initialize OpenRouter client for Grok 4
        self.openrouter_key = self.config.get('api_keys.openrouter')
        if self.openrouter_key:
            self.openrouter_client = openai.OpenAI(
                api_key=self.openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            )
        
        # Initialize components
        self.cost_tracker = CostTracker(self.budget, self.db_path)
        self.data_store = ChannelDataStore(self.db_path)
        self.constitution_manager = ConstitutionManager(self.db_path)
        self.memory_system = MemorySystem(self.db_path)
        
        # Tokenizer for cost estimation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Conversation threading - maintains proper conversation context
        self.conversation_threads = {}  # channel_id -> list of messages
        self.max_conversation_length = 20  # Keep last 20 messages per thread
        self.conversation_timeout = 3600  # 1 hour - reset conversation if inactive
        
        # Autonomous mode state
        self.autonomous_mode_active = False
        self.autonomous_tool_calls_used = 0
        self.autonomous_context_history = []
        
        # Function calling system
        self.available_functions = {
            'update_constitution': self._function_update_constitution,
            'query_memory': self._function_query_memory,
            'get_user_history': self._function_get_user_history,
            'store_observation': self._function_store_observation,
            'get_constitution': self._function_get_constitution,
            'think': self._function_think,
            'tag_user': self._function_tag_user,
            'message_user': self._function_message_user,
            'google_search': self._function_google_search,
            'update_importance_criteria': self._function_update_importance_criteria,
            'run_command': self._function_run_command,
            'create_script': self._function_create_script,
            'generate_image': self._function_generate_image,
            'get_budget': self._function_get_budget,
            'get_current_time': self._function_get_current_time,
            'send_message': self._function_send_message,
            'stop_autonomous_mode': self._function_stop_autonomous_mode,
            'summarize_context': self._function_summarize_context,
            'trigger_autonomous_mode': self._function_trigger_autonomous_mode,
            'edit_memory': self._function_edit_memory,
            'delete_memory': self._function_delete_memory
        }

    def _get_conversation_key(self, message) -> str:
        """Generate a unique key for conversation threading."""
        if isinstance(message.channel, discord.DMChannel):
            return f"dm_{message.author.id}"
        else:
            return f"channel_{message.channel.id}"
    
    def _add_to_conversation_thread(self, message, role: str, content: str):
        """Add a message to the conversation thread."""
        key = self._get_conversation_key(message)
        
        if key not in self.conversation_threads:
            self.conversation_threads[key] = []
        
        # Add the message with timestamp
        self.conversation_threads[key].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'user_id': message.author.id if hasattr(message, 'author') else None
        })
        
        # Keep only recent messages
        if len(self.conversation_threads[key]) > self.max_conversation_length:
            self.conversation_threads[key] = self.conversation_threads[key][-self.max_conversation_length:]
    
    def _get_conversation_thread(self, message) -> List[Dict[str, Any]]:
        """Get the conversation thread for this channel/user."""
        key = self._get_conversation_key(message)
        
        if key not in self.conversation_threads:
            return []
        
        # Check if conversation is too old and should be reset
        thread = self.conversation_threads[key]
        if thread:
            last_message_time = thread[-1]['timestamp']
            if (datetime.now() - last_message_time).total_seconds() > self.conversation_timeout:
                # Reset conversation if too old
                self.conversation_threads[key] = []
                return []
        
        return self.conversation_threads[key]
    
    def _build_conversation_messages(self, message, content: str, use_expensive: bool = False) -> List[Dict[str, Any]]:
        """Build the proper message array for API call with conversation context."""
        messages = []
        
        # Get existing conversation thread
        conversation_thread = self._get_conversation_thread(message)
        
        # Add conversation history
        for msg in conversation_thread:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        # Add current user message
        messages.append({
            'role': 'user',
            'content': content
        })
        
        return messages
    
    def _build_system_message(self, message) -> str:
        """Build the system message with context."""
        context_parts = [self.get_system_prompt()]
        
        # Add current context
        context_parts.append(f"\n## Current Context")
        context_parts.append(f"User: {message.author.display_name} (ID: {message.author.id})")
        if message.guild:
            context_parts.append(f"Server: {message.guild.name}")
            context_parts.append(f"Channel: #{getattr(message.channel, 'name', 'DM')}")
            
            # Check if user is in residency group
            member = message.guild.get_member(message.author.id)
            if member and any(role.name == self.residency_role_name for role in member.roles):
                context_parts.append("✨ User is part of the 25 residency group")
        else:
            context_parts.append("Channel: DM")
        
        # Add user history context for new conversations or returning users
        conversation_thread = self._get_conversation_thread(message)
        if len(conversation_thread) <= 1:  # New conversation or first message
            user_history = self.memory_system.get_user_interaction_history(message.author.id, 3)
            if user_history:
                context_parts.append("\n## User History Summary:")
                for interaction in user_history:
                    context_parts.append(f"- {interaction['timestamp'][:19]}: {interaction['content'][:100]}...")
        
        # AUTOMATIC SEMANTIC MEMORY RETRIEVAL - Always retrieve relevant memories
        if message.content.strip():
            # Clean the message content for semantic search
            search_content = message.content
            if self.user in message.mentions:
                search_content = search_content.replace(f'<@{self.user.id}>', '').strip()
            if search_content.startswith(self.prefix):
                search_content = search_content[len(self.prefix):].strip()
            
            if search_content and len(search_content) > 3:
                # Perform semantic search for relevant memories
                relevant_memories = self.memory_system.query_memories(
                    search_content, 
                    user_id=message.author.id,  # Focus on this user's memories
                    limit=5  # Top 5 most relevant
                )
                
                # Filter out very low similarity matches (< 0.3)
                high_relevance_memories = [m for m in relevant_memories if m.get('similarity', 0) >= 0.3]
                
                if high_relevance_memories:
                    context_parts.append("\n## Relevant Memories:")
                    for memory in high_relevance_memories:
                        similarity = memory.get('similarity', 0.0)
                        timestamp = memory['timestamp'][:19]
                        content_preview = memory['content'][:150]
                        memory_type = memory['memory_type']
                        context_parts.append(f"- [{similarity:.2f}] {timestamp} ({memory_type}): {content_preview}...")
                    
                    # Log memory retrieval for debugging
                    logger.info(f"🧠 Auto-retrieved {len(high_relevance_memories)} relevant memories for user query")
                    for i, memory in enumerate(high_relevance_memories[:2]):  # Show top 2 in logs
                        logger.info(f"  {i+1}. Similarity: {memory.get('similarity', 0.0):.3f} | {memory['content'][:80]}...")
        
        return "\n".join(context_parts)

    # Function call implementations
    def _function_update_constitution(self, new_constitution: str, reason: str = "") -> Dict[str, Any]:
        """Update the bot's constitution."""
        success = self.constitution_manager.update_constitution(new_constitution, reason)
        return {
            "success": success,
            "message": "Constitution updated successfully" if success else "Constitution update failed or no change needed"
        }
    
    def _function_query_memory(self, query: str, user_id: int = None, limit: int = 10) -> Dict[str, Any]:
        """Query the long-term memory system."""
        memories = self.memory_system.query_memories(query, user_id, limit=limit)
        
        # Detailed logging of memory retrieval
        logger.info(f"🧠 Memory Query: '{query}' (user_id={user_id}, limit={limit})")
        if memories:
            logger.info(f"📚 Retrieved {len(memories)} memories:")
            for i, memory in enumerate(memories[:3]):  # Show top 3
                similarity = memory.get('similarity', 0.0)
                content_preview = memory['content'][:100]
                logger.info(f"  {i+1}. Similarity: {similarity:.3f} | {content_preview}...")
        else:
            logger.info("📭 No memories found for query")
            
        return {
            "memories": memories,
            "count": len(memories),
            "query": query
        }
    
    def _function_get_user_history(self, user_id: int, limit: int = 20) -> Dict[str, Any]:
        """Get interaction history for a specific user."""
        history = self.memory_system.get_user_interaction_history(user_id, limit)
        return {
            "history": history,
            "count": len(history),
            "user_id": user_id
        }
    
    def _function_edit_memory(self, memory_id: str, new_content: str, reason: str = "") -> Dict[str, Any]:
        """Edit/update an existing memory with new content."""
        try:
            success = self.memory_system.edit_memory(memory_id, new_content, reason)
            logger.info(f"📝 Memory edit {'successful' if success else 'failed'}: {memory_id}")
            return {
                "success": success,
                "memory_id": memory_id,
                "message": f"Memory {'updated' if success else 'update failed'}: {reason}" if reason else f"Memory {'updated' if success else 'update failed'}"
            }
        except Exception as e:
            logger.error(f"Error editing memory: {e}")
            return {"error": str(e), "success": False, "memory_id": memory_id}
    
    def _function_delete_memory(self, memory_id: str, reason: str = "") -> Dict[str, Any]:
        """Delete a memory from the system."""
        try:
            success = self.memory_system.delete_memory(memory_id, reason)
            logger.info(f"🗑️ Memory deletion {'successful' if success else 'failed'}: {memory_id}")
            return {
                "success": success,
                "memory_id": memory_id,
                "message": f"Memory {'deleted' if success else 'deletion failed'}: {reason}" if reason else f"Memory {'deleted' if success else 'deletion failed'}"
            }
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"error": str(e), "success": False, "memory_id": memory_id}
    
    def _function_store_observation(self, observation: str, user_id: int = 0, channel_id: int = 0, tags: List[str] = None) -> Dict[str, Any]:
        """Store an observation or reflection in memory."""
        memory_id = self.memory_system.store_memory(
            observation, user_id, channel_id, 
            memory_type='observation', tags=tags or []
        )
        return {
            "success": bool(memory_id),
            "memory_id": memory_id,
            "message": "Observation stored successfully" if memory_id else "Failed to store observation"
        }
    
    def _function_get_constitution(self) -> Dict[str, Any]:
        """Get the current constitution."""
        constitution = self.constitution_manager.get_current_constitution()
        return {
            "constitution": constitution,
            "length": len(constitution)
        }
    
    def _function_think(self, thought: str) -> Dict[str, Any]:
        """Internal thinking/reflection function - not shown to users."""
        logger.info(f"Bot thinking: {thought[:100]}...")
        return {"thought_processed": True, "length": len(thought)}
    
    def _function_tag_user(self, user_id: int, context: str = "") -> Dict[str, Any]:
        """Tag/mention a user in responses."""
        return {
            "user_tag": f"<@{user_id}>",
            "user_id": user_id,
            "context": context
        }
    
    def _function_message_user(self, users: str, destination: str, message: str) -> Dict[str, Any]:
        """Unified messaging function - handles user messaging scenarios."""
        try:
            import json
            
            # Parse users JSON array
            try:
                if users.startswith('[') and users.endswith(']'):
                    user_list = json.loads(users)
                else:
                    user_list = [users] if users else []
            except:
                user_list = [users] if users else []
            
            # Handle current channel tagging
            if destination == "current":
                if user_list:
                    # Find users and create tags
                    user_tags = []
                    found_users = []
                    
                    for user_identifier in user_list:
                        # Try to find user by name or ID
                        if str(user_identifier).isdigit():
                            # It's a user ID
                            user_tags.append(f"<@{user_identifier}>")
                            found_users.append(user_identifier)
                        else:
                            # Try to find by name in guild members
                            target_guild_id = self.config.get('bot.target_guild_id')
                            guild = self.get_guild(target_guild_id) if target_guild_id else None
                            if guild:
                                for member in guild.members:
                                    if member.display_name.lower() == user_identifier.lower() or member.name.lower() == user_identifier.lower():
                                        user_tags.append(f"<@{member.id}>")
                                        found_users.append(member.display_name)
                                        break
                    
                    # Return tags to be included in response
                    tagged_message = f"{' '.join(user_tags)} {message}".strip()
                    return {
                        "success": True,
                        "user_count": len(found_users),
                        "destination": "current",
                        "action": "tag_in_response",
                        "tagged_message": tagged_message,
                        "message": f"Tagging {len(found_users)} user(s) in current channel"
                    }
                else:
                    # No tags, just general message
                    return {
                        "success": True,
                        "user_count": 0,
                        "destination": "current", 
                        "action": "general_message",
                        "tagged_message": message,
                        "message": "Sending general message to current channel"
                    }
                    
            elif destination == "dm":
                # Handle DM (simplified for now)
                return {
                    "success": True,
                    "destination": "dm",
                    "action": "dm_message",
                    "message": "DM functionality not yet implemented in this version",
                    "note": "Use current channel tagging instead"
                }
                
            else:
                # Handle specific channels (simplified for now)
                return {
                    "success": True,
                    "destination": destination,
                    "action": "channel_message", 
                    "message": f"Channel messaging to {destination} not yet implemented",
                    "note": "Use current channel tagging instead"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error in message_user function: {e}"
            }
    
    def _function_google_search(self, query: str) -> Dict[str, Any]:
        """Search web for strategic information (using Perplexity API)."""
        try:
            import requests
            import json
            
            # Use Perplexity API for better research results
            api_key = os.getenv('PERPLEXITY_API_KEY')
            if not api_key:
                # Fallback to basic web search if no Perplexity key
                return self._fallback_web_search(query)
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # Craft strategic query for Aug Lab needs
            enhanced_query = f"{query} latest trends strategies best practices 2024"
            
            payload = {
                'model': 'llama-3.1-sonar-small-128k-online',  # Cost-effective model
                'messages': [
                    {
                        'role': 'user',
                        'content': f"Search and summarize: {enhanced_query}. Focus on actionable insights, latest trends, and specific strategies."
                    }
                ],
                'max_tokens': 500,  # Keep short for budget
                'temperature': 0.1,
                'return_citations': True
            }
            
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                citations = data.get('citations', [])
                
                return {
                    "success": True,
                    "query": query,
                    "summary": content,
                    "citations": citations[:3],  # Limit citations for brevity
                    "source": "perplexity"
                }
            else:
                logger.error(f"Perplexity API error: {response.status_code}")
                return self._fallback_web_search(query)
                
        except Exception as e:
            logger.error(f"Error in Perplexity search: {e}")
            return self._fallback_web_search(query)
    
    def _fallback_web_search(self, query: str) -> Dict[str, Any]:
        """Fallback web search using free APIs."""
        try:
            import requests
            
            # Use DuckDuckGo as fallback
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(search_url, timeout=10)
            data = response.json()
            
            results = []
            if data.get('Abstract'):
                results.append({
                    'title': 'Summary',
                    'snippet': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', '')
                })
            
            for topic in data.get('RelatedTopics', [])[:2]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0],
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', '')
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "source": "duckduckgo_fallback"
            }
            
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "source": "fallback_failed"
            }
    
    def _function_update_importance_criteria(self, new_criteria: str, reason: str = "") -> Dict[str, Any]:
        """Update the importance criteria used by Haiku for message filtering."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Initialize table if needed
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS importance_criteria (
                    id INTEGER PRIMARY KEY,
                    criteria TEXT NOT NULL,
                    reason TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert new criteria
            cursor.execute("""
                INSERT INTO importance_criteria (criteria, reason)
                VALUES (?, ?)
            """, (new_criteria, reason))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated importance criteria: {reason}")
            
            return {
                "success": True,
                "new_criteria": new_criteria,
                "reason": reason,
                "message": "Importance criteria updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error updating importance criteria: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_importance_criteria(self) -> str:
        """Get current importance criteria for message filtering."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get latest criteria
            cursor.execute("""
                SELECT criteria FROM importance_criteria 
                ORDER BY updated_at DESC LIMIT 1
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0]
            else:
                # Default criteria
                return """
**Strategic Value (1-5)**: Does this help Aug Lab hit metrics? (social media, projects, connections)
**Technical Complexity (1-5)**: Needs advanced reasoning/research?
**Social Connection (1-5)**: Helps members network/collaborate?
**Project Help (1-5)**: Assists with project completion?

**DECISION RULE**: Use expensive model if:
- Strategic value ≥ 4 (directly helps metrics)
- Technical complexity ≥ 4 AND any other factor ≥ 3
- Multiple factors ≥ 3 (compound benefit)

**BUDGET PRIORITY**: Conserve compute for maximum Aug Lab impact.
                """.strip()
                
        except Exception as e:
            logger.error(f"Error getting importance criteria: {e}")
            return "Use expensive model sparingly - conserve budget for maximum Aug Lab impact."
    
    def _function_generate_image(self, prompt: str, quality: str = "medium", size: str = "1024x1024") -> Dict[str, Any]:
        """Generate an image using GPT-4o's native image generation. Cost varies by quality: low (~$0.01), medium (~$0.04), high (~$0.17)."""
        try:
            # Check budget first
            estimated_tokens = 1200  # Rough estimate for medium quality
            if quality == "high":
                estimated_tokens = 4200
            elif quality == "low":
                estimated_tokens = 300
                
            if not self.cost_tracker.can_afford('gpt4o', estimated_tokens):
                return {
                    "success": False,
                    "error": "Insufficient budget for image generation",
                    "budget_remaining": self.cost_tracker.get_remaining_budget(),
                    "estimated_cost": estimated_tokens * 0.00004  # Rough estimate
                }
            
            # Validate quality and size
            valid_qualities = ["low", "medium", "high"]
            valid_sizes = ["1024x1024", "1024x1536", "1536x1024"]
            
            if quality not in valid_qualities:
                quality = "medium"
            if size not in valid_sizes:
                size = "1024x1024"
            
            # Generate the image
            response = self.openai_client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            # Calculate accurate cost based on GPT-4o pricing
            input_tokens = len(prompt.split()) * 1.25  # Estimate for text input
            if quality == "low":
                output_tokens = 272  # Low quality token count
                base_cost = 0.011
            elif quality == "high":
                output_tokens = 4160  # High quality token count  
                base_cost = 0.166
            else:  # medium
                output_tokens = 1056  # Medium quality token count
                base_cost = 0.042
                
            # Add text input cost: $5.00 per 1M tokens
            actual_cost = base_cost + (input_tokens / 1000000) * 5.0
            
            # Record the usage
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model='gpt-image-1',
                input_tokens=int(input_tokens),
                output_tokens=output_tokens,
                cost=actual_cost,
                action='image_generation_function',
                user_id=0,  # System call
                channel_id=0
            ))
            
            return {
                "success": True,
                "image_url": response.data[0].url,
                "prompt": prompt,
                "quality": quality,
                "size": size,
                "cost": round(actual_cost, 3),
                "budget_remaining": round(self.cost_tracker.get_remaining_budget(), 2),
                "message": f"Generated {quality} quality image for ${actual_cost:.3f}"
            }
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }
    
    def _function_run_command(self, command: str, timeout: int = 30, language: str = "bash") -> Dict[str, Any]:
        """Execute command line tools safely in Docker sandbox for strategic information gathering."""
        try:
            # Import secure executor
            from secure_command_executor import secure_run_command
            
            # Execute in Docker sandbox
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                secure_run_command(command, language, timeout)
            )
            
            logger.info(f"Secure command executed: {command[:50]}... -> {result['success']}")
            return result
            
        except ImportError:
            # Fallback to disabled state if Docker not available
            logger.warning("Secure executor not available - Docker required for command execution")
            return {
                "success": False,
                "command": command,
                "error": "Docker sandboxing required but not available. Install Docker to enable secure command execution."
            }
        except Exception as e:
            logger.error(f"Error in secure command execution: {e}")
            return {
                "success": False,
                "command": command,
                "error": f"Secure execution error: {e}"
            }
    
    def _function_create_script(self, filename: str, content: str, language: str = "bash") -> Dict[str, Any]:
        """Create and execute scripts for complex Aug Lab tasks."""
        try:
            import tempfile
            import subprocess
            import os
            
            # Security: limit script types and content
            allowed_languages = {'bash', 'python', 'javascript', 'playwright'}
            
            if language not in allowed_languages:
                return {
                    "success": False,
                    "error": f"Language '{language}' not allowed. Use: {', '.join(allowed_languages)}"
                }
            
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                if language == 'bash':
                    f.write("#!/bin/bash\nset -e\n" + content)
                    interpreter = 'bash'
                elif language == 'python':
                    f.write(content)
                    interpreter = 'python3'
                elif language == 'javascript':
                    f.write(content)
                    interpreter = 'node'
                elif language == 'playwright':
                    # Playwright script for web automation
                    playwright_template = f"""
const {{ chromium }} = require('playwright');

(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();
    
    try {{
        {content}
    }} catch (error) {{
        console.error('Error:', error.message);
    }} finally {{
        await browser.close();
    }}
}})();
"""
                    f.write(playwright_template)
                    interpreter = 'node'
                
                script_path = f.name
            
            # Make executable
            os.chmod(script_path, 0o755)
            
            # Execute script with timeout
            result = subprocess.run(
                [interpreter, script_path],
                capture_output=True,
                text=True,
                timeout=60,  # Longer timeout for scripts
                cwd='/tmp'
            )
            
            # Clean up
            os.unlink(script_path)
            
            output = result.stdout.strip()
            error = result.stderr.strip()
            
            # Limit output for budget
            max_output = 1500
            if len(output) > max_output:
                output = output[:max_output] + f"\n... (truncated from {len(output)} chars)"
            
            return {
                "success": result.returncode == 0,
                "language": language,
                "filename": filename,
                "output": output,
                "error": error if error else None,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script execution timed out (60s limit)"
            }
        except Exception as e:
            logger.error(f"Error creating/executing script: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _function_get_budget(self) -> Dict[str, Any]:
        """Get current budget status and spending breakdown for strategic decision making."""
        try:
            report = self.cost_tracker.get_spending_report()
            percentage_used = (report['total_spent'] / report['budget']) * 100
            
            return {
                "success": True,
                "budget": {
                    "total_budget": report['budget'],
                    "spent": report['total_spent'],
                    "remaining": report['remaining'],
                    "percentage_used": round(percentage_used, 1),
                    "by_model": report.get('by_model', {}),
                    "can_afford_expensive": self.cost_tracker.can_afford('sonnet', 2000),
                    "can_afford_cheap": self.cost_tracker.can_afford('haiku', 1000),
                    "warning": "Low budget!" if report['remaining'] < 10 else None
                },
                "message": f"Budget: ${report['remaining']:.2f} remaining of ${report['budget']:.2f} ({percentage_used:.1f}% used)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting budget info: {str(e)}"
            }

    def _function_get_current_time(self) -> Dict[str, Any]:
        """Get current time and calculate days until MIT Media Lab presentation (Aug 23rd)."""
        try:
            from datetime import datetime, timezone
            
            now = datetime.now(timezone.utc)
            
            # MIT Media Lab presentation deadline
            presentation_date = datetime(2025, 8, 23, 12, 0, 0, tzinfo=timezone.utc)  # Assume noon EST
            
            days_until_presentation = (presentation_date - now).days
            hours_until_presentation = (presentation_date - now).total_seconds() / 3600
            
            # Calculate urgency level
            if days_until_presentation < 0:
                urgency = "DEADLINE_PASSED"
                time_status = "Presentation deadline has passed!"
            elif days_until_presentation <= 7:
                urgency = "CRITICAL"
                time_status = f"⚠️ CRITICAL: Only {days_until_presentation} days until presentation!"
            elif days_until_presentation <= 14:
                urgency = "HIGH"
                time_status = f"⚡ HIGH URGENCY: {days_until_presentation} days until presentation"
            elif days_until_presentation <= 30:
                urgency = "MEDIUM"
                time_status = f"📅 {days_until_presentation} days until presentation"
            else:
                urgency = "LOW"
                time_status = f"📅 {days_until_presentation} days until presentation"
            
            return {
                "success": True,
                "time": {
                    "current_utc": now.isoformat(),
                    "current_est": now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "presentation_date": "August 23rd, 2025",
                    "days_until_presentation": days_until_presentation,
                    "hours_until_presentation": round(hours_until_presentation, 1),
                    "urgency_level": urgency,
                    "time_status": time_status,
                    "week_number": now.isocalendar()[1],  # Week of year
                    "is_weekend": now.weekday() >= 5,
                    "month_name": now.strftime("%B"),
                    "day_name": now.strftime("%A")
                },
                "message": f"Current time: {now.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} | {time_status}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting time info: {str(e)}"
            }

    def _function_send_message(self, channel_id: int, message: str, channel_name: str = "") -> Dict[str, Any]:
        """Send a message to a specific channel (autonomous mode only)."""
        if not self.autonomous_mode_active:
            return {
                "success": False,
                "error": "send_message can only be used in autonomous mode",
                "message": "This function is restricted to autonomous operations"
            }
            
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                return {
                    "success": False,
                    "error": f"Channel {channel_id} not found",
                    "message": f"Could not find channel {channel_name or channel_id}"
                }
            
            # Store the message task for execution (we can't use async in sync function)
            self._pending_autonomous_message = {
                "channel": channel,
                "message": message[:2000],  # Discord limit
                "timestamp": datetime.now()
            }
            
            return {
                "success": True,
                "channel_id": channel_id,
                "channel_name": channel.name if hasattr(channel, 'name') else 'DM',
                "message_length": len(message),
                "message": f"Message queued for #{channel.name if hasattr(channel, 'name') else 'DM'}: {message[:100]}..."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error queueing message: {str(e)}"
            }

    def _function_stop_autonomous_mode(self, reason: str = "Task completed") -> Dict[str, Any]:
        """Stop the autonomous mode and return control."""
        if not self.autonomous_mode_active:
            return {
                "success": False,
                "message": "Autonomous mode is not currently active"
            }
        
        self.autonomous_mode_active = False
        tool_calls_used = self.autonomous_tool_calls_used
        self.autonomous_tool_calls_used = 0
        
        # Store final observation
        self.memory_system.store_memory(
            f"Autonomous mode session completed. Reason: {reason}. Used {tool_calls_used} tool calls.",
            user_id=0,
            channel_id=0,
            memory_type='observation',
            tags=['autonomous', 'session_end']
        )
        
        return {
            "success": True,
            "reason": reason,
            "tool_calls_used": tool_calls_used,
            "message": f"Autonomous mode stopped: {reason} (Used {tool_calls_used} tool calls)"
        }

    def _function_trigger_autonomous_mode(self, reason: str = "AI-triggered strategic analysis") -> Dict[str, Any]:
        """Trigger autonomous mode for strategic intelligence operations."""
        if self.autonomous_mode_active:
            return {
                "success": False,
                "message": "Autonomous mode is already active"
            }
        
        if self.cost_tracker.get_remaining_budget() < 1.0:
            return {
                "success": False,
                "message": f"Insufficient budget for autonomous mode (${self.cost_tracker.get_remaining_budget():.2f} remaining)"
            }
        
        # Schedule autonomous thinking to run asynchronously
        import asyncio
        asyncio.create_task(self.autonomous_thinking())
        
        return {
            "success": True,
            "reason": reason,
            "message": f"Autonomous mode triggered: {reason}. Strategic operations will begin shortly."
        }

    def _function_summarize_context(self, context_type: str = "all") -> Dict[str, Any]:
        """Summarize various types of context to manage tokens efficiently."""
        if context_type == "all":
            chat_summary = self._summarize_chat_history()
            tool_summary = self._summarize_tool_results()
            return {
                "chat_summary": chat_summary,
                "tool_summary": tool_summary,
                "total_length": len(chat_summary) + len(tool_summary)
            }
        elif context_type == "chat_history":
            summary = self._summarize_chat_history()
            return {
                "summary": summary,
                "length": len(summary)
            }
        elif context_type == "tool_results":
            summary = self._summarize_tool_results()
            return {
                "summary": summary,
                "length": len(summary)
            }
        else:
            return {
                "error": f"Unknown context type: {context_type}",
                "available_types": ["all", "chat_history", "tool_results"]
            }
    
    def _summarize_chat_history(self) -> str:
        """Create a summary of recent chat history."""
        try:
            # Get recent messages from channels
            recent_messages = []
            if self.target_guild_id:
                guild = self.get_guild(self.target_guild_id)
                if guild:
                    for channel in guild.text_channels:
                        history = self.data_store.get_channel_history(channel.id, limit=10)
                        recent_messages.extend(history[-5:])  # Last 5 from each channel
            
            if not recent_messages:
                return "No recent chat activity"
            
            # Sort by timestamp
            recent_messages.sort(key=lambda x: x['timestamp'])
            
            # Create summary
            summary_parts = []
            for msg in recent_messages[-10:]:  # Last 10 overall
                author = msg['author_name']
                content = msg['content'][:100]
                channel = msg['channel_name']
                summary_parts.append(f"#{channel} - {author}: {content}")
            
            return "Recent activity:\n" + "\n".join(summary_parts)
        except Exception as e:
            return f"Error summarizing chat: {str(e)}"

    def _summarize_tool_results(self) -> str:
        """Create a summary of recent tool call results."""
        try:
            # Get recent tool results from autonomous context history
            tool_results = [
                item for item in self.autonomous_context_history 
                if item.get('type') == 'tool_result'
            ]
            
            if not tool_results:
                return "No recent tool calls"
            
            # Summarize recent tools
            summary_parts = []
            for result in tool_results[-10:]:  # Last 10 tool calls
                func_name = result.get('function', 'unknown')
                success = result.get('success', False)
                summary = result.get('summary', result.get('content', ''))[:100]
                summary_parts.append(f"{func_name}: {'✅' if success else '❌'} {summary}")
            
            return "Recent tools:\n" + "\n".join(summary_parts)
        except Exception as e:
            return f"Error summarizing tools: {str(e)}"

    def get_system_prompt(self) -> str:
        """Get the core system prompt with function calling instructions."""
        constitution = self.constitution_manager.get_current_constitution()
        
        # Get the main system prompt from config
        main_prompt = self.config.get('system_prompt.main_prompt', '')
        
        # Handle both string and array formats for backward compatibility
        if isinstance(main_prompt, list):
            main_prompt = '\n'.join(main_prompt)
        
        # Add constitution to the prompt
        return f"""{main_prompt}

## Current Constitution
{constitution}"""

    def parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse XML-style command tags from AI response."""
        function_calls = []
        
        # Parse different command patterns
        patterns = {
            'think': r'<think>(.*?)</think>',
            'query_memory': r'<query_memory>(.*?)</query_memory>', 
            'get_user_history': r'<get_user_history user_id="(\d+)".*?>(.*?)</get_user_history>',
            'store_observation': r'<store_observation user_id="(\d+)" channel_id="(\d+)" tags="([^"]*)">(.*?)</store_observation>',
            'update_constitution': r'<update_constitution reason="([^"]*)">(.*?)</update_constitution>',
            'get_constitution': r'<get_constitution.*?>(.*?)</get_constitution>',
            'tag_user': r'<tag_user user_id="(\d+)".*?>(.*?)</tag_user>',
            'message_user': r'<message_user users="([^"]*)" destination="([^"]*)">(.*?)</message_user>',
            'google_search': r'<google_search>(.*?)</google_search>',
            'run_command': r'<run_command language="([^"]*)">(.*?)</run_command>',
            'generate_image': r'<generate_image(?:\s+quality="([^"]*)")?(?:\s+size="([^"]*)")?>(.*?)</generate_image>',
            'update_importance_criteria': r'<update_importance_criteria reason="([^"]*)">(.*?)</update_importance_criteria>',
            'get_budget': r'<get_budget.*?>(.*?)</get_budget>',
            'get_current_time': r'<get_current_time.*?>(.*?)</get_current_time>',
            'send_message': r'<send_message channel_id="(\d+)"(?:\s+channel_name="([^"]*)")?>(.*?)</send_message>',
            'stop_autonomous_mode': r'<stop_autonomous_mode(?:\s+reason="([^"]*)")?>(.*?)</stop_autonomous_mode>',
            'summarize_context': r'<summarize_context(?:\s+context_type="([^"]*)")?>(.*?)</summarize_context>',
            'trigger_autonomous_mode': r'<trigger_autonomous_mode(?:\s+reason="([^"]*)")?>(.*?)</trigger_autonomous_mode>',
            'edit_memory': r'<edit_memory memory_id="([^"]*)" reason="([^"]*)">(.*?)</edit_memory>',
            'delete_memory': r'<delete_memory memory_id="([^"]*)" reason="([^"]*)">(.*?)</delete_memory>'
        }
        
        for func_name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                if func_name == 'think':
                    function_calls.append({
                        'function': 'think',
                        'parameters': {'thought': match.strip()}
                    })
                elif func_name == 'query_memory':
                    function_calls.append({
                        'function': 'query_memory', 
                        'parameters': {'query': match.strip()}
                    })
                elif func_name == 'get_user_history':
                    user_id, limit = match
                    function_calls.append({
                        'function': 'get_user_history',
                        'parameters': {'user_id': int(user_id), 'limit': int(limit.strip()) if limit.strip() else 20}
                    })
                elif func_name == 'store_observation':
                    user_id, channel_id, tags, observation = match
                    function_calls.append({
                        'function': 'store_observation',
                        'parameters': {
                            'observation': observation.strip(),
                            'user_id': int(user_id),
                            'channel_id': int(channel_id), 
                            'tags': tags.split(',') if tags else []
                        }
                    })
                elif func_name == 'update_constitution':
                    reason, new_constitution = match
                    function_calls.append({
                        'function': 'update_constitution',
                        'parameters': {
                            'new_constitution': new_constitution.strip(),
                            'reason': reason
                        }
                    })
                elif func_name == 'get_constitution':
                    function_calls.append({
                        'function': 'get_constitution',
                        'parameters': {}
                    })
                elif func_name == 'tag_user':
                    user_id, context = match
                    function_calls.append({
                        'function': 'tag_user',
                        'parameters': {
                            'user_id': int(user_id),
                            'context': context.strip()
                        }
                    })
                elif func_name == 'message_user':
                    users, destination, message_content = match
                    function_calls.append({
                        'function': 'message_user',
                        'parameters': {
                            'users': users.strip(),
                            'destination': destination.strip(),
                            'message': message_content.strip()
                        }
                    })
                elif func_name == 'google_search':
                    function_calls.append({
                        'function': 'google_search',
                        'parameters': {
                            'query': match.strip()
                        }
                    })
                elif func_name == 'run_command':
                    language, command = match
                    function_calls.append({
                        'function': 'run_command',
                        'parameters': {
                            'command': command.strip(),
                            'language': language.strip()
                        }
                    })
                elif func_name == 'generate_image':
                    quality, size, prompt = match
                    function_calls.append({
                        'function': 'generate_image',
                        'parameters': {
                            'prompt': prompt.strip(),
                            'quality': quality.strip() if quality else 'medium',
                            'size': size.strip() if size else '1024x1024'
                        }
                    })
                elif func_name == 'update_importance_criteria':
                    reason, new_criteria = match
                    function_calls.append({
                        'function': 'update_importance_criteria',
                        'parameters': {
                            'new_criteria': new_criteria.strip(),
                            'reason': reason
                        }
                    })
                elif func_name == 'get_budget':
                    function_calls.append({
                        'function': 'get_budget',
                        'parameters': {}
                    })
                elif func_name == 'get_current_time':
                    function_calls.append({
                        'function': 'get_current_time',
                        'parameters': {}
                    })
                elif func_name == 'send_message':
                    channel_id, channel_name, message_content = match
                    function_calls.append({
                        'function': 'send_message',
                        'parameters': {
                            'channel_id': int(channel_id),
                            'message': message_content.strip(),
                            'channel_name': channel_name.strip() if channel_name else ''
                        }
                    })
                elif func_name == 'stop_autonomous_mode':
                    reason, _ = match
                    function_calls.append({
                        'function': 'stop_autonomous_mode',
                        'parameters': {
                            'reason': reason.strip() if reason else 'Task completed'
                        }
                    })
                elif func_name == 'summarize_context':
                    context_type, _ = match
                    function_calls.append({
                        'function': 'summarize_context',
                        'parameters': {
                            'context_type': context_type.strip() if context_type else 'all'
                        }
                    })
                elif func_name == 'trigger_autonomous_mode':
                    reason, _ = match
                    function_calls.append({
                        'function': 'trigger_autonomous_mode',
                        'parameters': {
                            'reason': reason.strip() if reason else 'AI-triggered strategic analysis'
                        }
                    })
                elif func_name == 'edit_memory':
                    memory_id, reason, new_content = match
                    function_calls.append({
                        'function': 'edit_memory',
                        'parameters': {
                            'memory_id': memory_id.strip(),
                            'reason': reason.strip(),
                            'new_content': new_content.strip()
                        }
                    })
                elif func_name == 'delete_memory':
                    memory_id, reason, _ = match
                    function_calls.append({
                        'function': 'delete_memory',
                        'parameters': {
                            'memory_id': memory_id.strip(),
                            'reason': reason.strip()
                        }
                    })
                    
        return function_calls

    async def execute_function_calls(self, function_calls: List[Dict[str, Any]]) -> str:
        """Execute function calls and return results."""
        results = []
        
        for call in function_calls:
            func_name = call.get('function')
            params = call.get('parameters', {})
            
            if func_name in self.available_functions:
                try:
                    result = self.available_functions[func_name](**params)
                    results.append(f"✅ **{func_name}**: {json.dumps(result, indent=2)}")
                except Exception as e:
                    logger.error(f"Error executing function {func_name}: {e}")
                    results.append(f"❌ **{func_name}**: Error - {str(e)}")
            else:
                results.append(f"❌ **{func_name}**: Function not found")
        
        return "\n\n".join(results) if results else ""
        
    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        logger.info(f'Budget remaining: ${self.cost_tracker.get_remaining_budget():.2f}')
        
        # Start background tasks (avoid double-start error)
        if not self.update_member_cache.is_running():
            self.update_member_cache.start()
        if not self.budget_monitor.is_running():
            self.budget_monitor.start()
        if not self.autonomous_investigation.is_running():
            self.autonomous_investigation.start()
        
        # Update channel tracking status
        await self.update_channel_tracking()
        
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="the Augmentation Lab 🧪"
            )
        )

    async def on_message(self, message):
        """Handle all incoming messages."""
        # Don't respond to bots
        if message.author.bot:
            return
            
        # Check guild restrictions (if enabled)
        if self.restricted_mode and message.guild:
            if self.allowed_guilds and message.guild.id not in self.allowed_guilds:
                # Bot is in restricted mode and this guild is not allowed
                return
            
        # Check if user has the required role (only in guilds, not DMs)
        # Made more permissive: allow if no residency role exists or user has it
        if message.guild:
            member = message.guild.get_member(message.author.id)
            if member:
                # Check if the residency role exists in this guild
                role_exists = any(role.name == self.residency_role_name for role in message.guild.roles)
                if role_exists:
                    # If role exists, user must have it
                    if not any(role.name == self.residency_role_name for role in member.roles):
                        return
                # If role doesn't exist, allow anyone to use the bot
            
        # Store all messages for future reference
        self.data_store.store_message(message)
        
        # Process commands first - if it's a valid command, don't send to AI
        await self.process_commands(message)
        
        # Check if this was a valid command
        if message.content.startswith(self.prefix):
            # Extract potential command name
            parts = message.content[len(self.prefix):].split()
            if parts:
                command_name = parts[0].lower()
                # List of valid command names
                valid_commands = ['budget', 'constitution', 'memory_query', 'user_history', 
                                'memory_stats', 'dm_residency', 'search', 'generate_image', 'help',
                                'autonomous_think', 'autonomous_status', 'guilds', 'models', 'switch_model',
                                'response_mode']
                if command_name in valid_commands:
                    return  # Don't send valid commands to AI
        
        # Determine if we should monitor for memory (always if enabled)
        should_monitor_memory = self.always_monitor_memory and (
            # Always monitor DMs
            isinstance(message.channel, discord.DMChannel) or
            # Monitor all messages in target guild (Aug Lab) if set
            (message.guild and self.target_guild_id and message.guild.id == self.target_guild_id) or
            # If no target guild set, monitor all guilds (for initial setup)
            (message.guild and not self.target_guild_id) or
            # Monitor if contains Aug Lab keywords
            any(word in message.content.lower() for word in ['auglab', 'lab', 'project', 'presentation', 'social media', 'twitter', 'x.com', 'viral'])
        )
        
        # Determine if we should respond based on response_mode
        should_respond = self._should_respond_to_message(message)
        logger.info(f"🎯 Message routing: should_monitor_memory={should_monitor_memory}, should_respond={should_respond}")
        
        # Always monitor for memory if enabled
        if should_monitor_memory:
            logger.info("🧠 Calling memory_monitor")
            await self.memory_monitor(message)
        
        # Only respond if configured to do so
        if should_respond:
            logger.info("📡 Calling strategic_monitor")
            await self.strategic_monitor(message)
        else:
            logger.info("❌ Not responding - response_mode criteria not met")

    async def strategic_monitor(self, message):
        """Haiku monitors messages and decides whether to respond and escalate to Sonnet."""
        logger.info(f"📡 Strategic monitor triggered for {message.author.display_name}: '{message.content[:50]}...'")
        
        if self.cost_tracker.get_remaining_budget() <= 0:
            logger.warning("📡 Strategic monitor: Budget exhausted, skipping")
            return  # Don't even monitor if no budget
            
        # Clean message content
        content = message.content
        if self.user in message.mentions:
            content = content.replace(f'<@{self.user.id}>', '').strip()
        if content.startswith(self.prefix):
            content = content[len(self.prefix):].strip()
            
        # Always respond to direct mentions, even if content is short
        is_direct_mention = self.user in message.mentions
        
        if (not content or len(content) < 3) and not is_direct_mention:
            logger.info(f"📡 Strategic monitor: Content too short ('{content}'), skipping")
            return  # Skip very short messages unless it's a direct mention
            
        try:
            # Use Haiku to assess strategic value and determine action
            base_monitoring_prompt = self.config.get('system_prompt.monitoring_prompt', '')
            
            # Handle both string and array formats for backward compatibility
            if isinstance(base_monitoring_prompt, list):
                base_monitoring_prompt = '\n'.join(base_monitoring_prompt)
            
            # Handle very short content for direct mentions
            if is_direct_mention and (not content or len(content) < 3):
                content_for_prompt = content or "brief greeting"
            else:
                content_for_prompt = content
            
            monitoring_prompt = f"""{base_monitoring_prompt}

## MESSAGE TO ASSESS
Message: "{content_for_prompt}"
Author: {message.author.display_name}
Channel: #{message.channel.name if hasattr(message.channel, 'name') else 'DM'}
Direct mention: {"Yes" if self.user in message.mentions else "No"}"""

            input_tokens = len(self.tokenizer.encode(monitoring_prompt))
            
            if not self.cost_tracker.can_afford('haiku', input_tokens + 150):
                return  # Skip if can't afford monitoring
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model=self.cheap_model_config.get('model', 'claude-3-haiku-20240307'),
                    max_tokens=200,
                    messages=[{"role": "user", "content": monitoring_prompt}]
                )
            )
            
            output_tokens = len(self.tokenizer.encode(response.content[0].text))
            cost = self.cost_tracker.calculate_cost('haiku', input_tokens, output_tokens)
            
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model='haiku',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                action='strategic_monitoring',
                user_id=message.author.id,
                channel_id=message.channel.id
            ))
            
            # Parse Haiku's decision
            try:
                import json
                logger.info(f"📡 Strategic monitor response: '{response.content[0].text[:200]}...'")
                decision = json.loads(response.content[0].text)
                logger.info(f"📡 Parsed decision: {decision}")
                
                # Store strategic observation if valuable
                if decision.get('strategic_value', 0) >= 3:
                    logger.info(f"📡 Storing strategic observation (value: {decision.get('strategic_value')})")
                    self.memory_system.store_memory(
                        f"Strategic observation: {content[:200]} - {decision.get('reasoning', '')}",
                        message.author.id,
                        message.channel.id,
                        memory_type='observation',
                        tags=['strategy', 'monitoring', decision.get('response_type', 'general')]
                    )
                
                # Always respond to direct mentions, regardless of strategic assessment
                if self.user in message.mentions:
                    logger.info("📡 DIRECT MENTION DETECTED - overriding strategic decision, will respond")
                    if decision.get('use_sonnet', False) and self.cost_tracker.can_afford('sonnet', 2000):
                        # Escalate to Sonnet for strategic response
                        logger.info("📡 Escalating to Sonnet")
                        await self.handle_ai_response(message, force_expensive=True)
                    elif self.cost_tracker.can_afford('haiku', 1000):
                        # Haiku can handle this response
                        logger.info("📡 Using Haiku for response")
                        await self.handle_ai_response(message, force_expensive=False)
                    else:
                        logger.warning("📡 Wanted to respond but insufficient budget")
                # For non-mentions, follow strategic decision
                elif decision.get('should_respond', False):
                    logger.info(f"📡 Decision: SHOULD RESPOND - use_sonnet: {decision.get('use_sonnet', False)}")
                    if decision.get('use_sonnet', False) and self.cost_tracker.can_afford('sonnet', 2000):
                        # Escalate to Sonnet for strategic response
                        logger.info("📡 Escalating to Sonnet")
                        await self.handle_ai_response(message, force_expensive=True)
                    elif self.cost_tracker.can_afford('haiku', 1000):
                        # Haiku can handle this response
                        logger.info("📡 Using Haiku for response")
                        await self.handle_ai_response(message, force_expensive=False)
                    else:
                        logger.warning("📡 Wanted to respond but insufficient budget")
                else:
                    logger.info(f"📡 Decision: NO RESPONSE needed - reasoning: {decision.get('reasoning', 'none')}")
                        
            except json.JSONDecodeError as e:
                logger.error(f"📡 JSON parsing failed: {e} - Raw response: '{response.content[0].text}'")
                # If JSON parsing fails, default to conservative approach
                if self.user in message.mentions and self.cost_tracker.can_afford('haiku', 1000):
                    logger.info("📡 JSON failed, but direct mention detected - responding")
                    await self.handle_ai_response(message, force_expensive=False)
                    
        except Exception as e:
            logger.error(f"Error in strategic monitoring: {e}")
            # Fallback: only respond to direct mentions
            if self.user in message.mentions:
                await self.handle_ai_response(message, force_expensive=False)

    async def handle_ai_response(self, message, force_expensive=None):
        """Handle AI responses with intelligent model selection."""
        logger.info(f"🤖 Starting AI response for {message.author.display_name}: '{message.content[:100]}...'")
        
        if self.cost_tracker.get_remaining_budget() <= 0:
            logger.warning("❌ Budget exhausted!")
            await message.reply("⚠️ Budget exhausted! Cannot process AI requests until reset.")
            return
            
        # Clean message content
        content = message.content
        if self.user in message.mentions:
            content = content.replace(f'<@{self.user.id}>', '').strip()
        if content.startswith(self.prefix):
            content = content[len(self.prefix):].strip()
            
        if not content:
            content = "Hello!"
            
        logger.info(f"🧹 Cleaned content: '{content}'")
            
        async with message.channel.typing():
            # Determine model selection
            if force_expensive is not None:
                # Strategic monitor has already decided
                use_expensive = force_expensive and self.cost_tracker.can_afford('sonnet', 2000)
                logger.info(f"📊 Model selection forced: {'Sonnet' if use_expensive else 'Haiku'}")
            else:
                # Use original importance assessment
                logger.info("📊 Assessing message importance...")
                importance_assessment = await self.assess_importance(message, content)
                use_expensive = importance_assessment['use_expensive_model'] and self.cost_tracker.can_afford('sonnet', 2000)
                logger.info(f"📊 Importance assessment: {importance_assessment}")
                logger.info(f"📊 Using model: {'Sonnet' if use_expensive else 'Haiku'}")
            
            logger.info("🎯 Generating response...")
            response = await self.generate_response(message, content, use_expensive=use_expensive)
            logger.info(f"📝 Generated response ({len(response) if response else 0} chars): '{response[:200] if response else 'None'}...'")
                
        if response:
            logger.info("📤 Sending response to Discord...")
            try:
                # Split long messages
                if len(response) > 2000:
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    logger.info(f"📝 Splitting into {len(chunks)} chunks")
                    for i, chunk in enumerate(chunks):
                        await message.reply(chunk)
                        logger.info(f"✅ Sent chunk {i+1}/{len(chunks)}")
                else:
                    await message.reply(response)
                    logger.info("✅ Response sent successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to send response: {e}")
        else:
            logger.warning("❌ No response generated!")
        
    async def assess_importance(self, message, content) -> Dict[str, Any]:
        """Use Haiku to assess if a message needs expensive model treatment."""
        try:
            # Get current importance criteria (can be updated by Sonnet)
            importance_criteria = self.get_importance_criteria()
            
            assessment_prompt = f"""You are a message importance filter for the Aug Lab Discord bot. Your job is to decide if this message needs the expensive Sonnet model or if Haiku can handle it.

## CURRENT IMPORTANCE CRITERIA
{importance_criteria}

## MESSAGE TO ASSESS
Message: "{content}"
Author: {message.author.display_name}
Channel: #{message.channel.name if hasattr(message.channel, 'name') else 'DM'}
Context: Aug Lab summer program focused on hitting specific metrics (social media virality, project completion, member satisfaction)

## ASSESSMENT
Rate each factor (1-5) and decide:

Respond with JSON only:
{{
    "strategic_value": X,
    "technical_complexity": X, 
    "social_connection": X,
    "project_help": X,
    "use_expensive_model": true/false,
    "reasoning": "brief explanation"
}}

Remember: Budget is LIMITED. Only use expensive model when truly needed for Aug Lab success."""

            input_tokens = len(self.tokenizer.encode(assessment_prompt))
            
            if not self.cost_tracker.can_afford('haiku', input_tokens + 100):
                return {'use_expensive_model': False, 'reasoning': 'Budget constraint'}
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model=self.cheap_model_config.get('model', 'claude-3-haiku-20240307'),
                    max_tokens=200,
                    messages=[{"role": "user", "content": assessment_prompt}]
                )
            )
            
            output_tokens = len(self.tokenizer.encode(response.content[0].text))
            cost = self.cost_tracker.calculate_cost('haiku', input_tokens, output_tokens)
            
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model='haiku',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                action='importance_assessment',
                user_id=message.author.id,
                channel_id=message.channel.id
            ))
            
            # Parse JSON response
            try:
                import json
                result = json.loads(response.content[0].text)
                return result
            except:
                # Fallback if JSON parsing fails
                return {'use_expensive_model': True, 'reasoning': 'Parse error, using expensive model'}
                
        except Exception as e:
            logger.error(f"Error in importance assessment: {e}")
            return {'use_expensive_model': False, 'reasoning': 'Error occurred'}

    async def generate_response(self, message, content, use_expensive=False) -> Optional[str]:
        """Generate AI response using appropriate model with conversation threading."""
        try:
            model_config = self.expensive_model_config if use_expensive else self.cheap_model_config
            model_name = model_config.get('name', 'haiku')
            model = model_config.get('model', 'claude-3-haiku-20240307')
            
            # Log conversation threading info
            conversation_thread = self._get_conversation_thread(message)
            logger.info(f"💬 Conversation thread for {message.author.display_name}: {len(conversation_thread)} messages")
            
            # Add user message to conversation thread
            self._add_to_conversation_thread(message, 'user', content)
            
            # Build system message with context
            system_message = self._build_system_message(message)
            
            # Build conversation messages
            conversation_messages = self._build_conversation_messages(message, content, use_expensive)
            
            # Estimate tokens for both system and conversation messages
            system_tokens = len(self.tokenizer.encode(system_message))
            conversation_tokens = sum(len(self.tokenizer.encode(msg['content'])) for msg in conversation_messages)
            total_input_tokens = system_tokens + conversation_tokens
            
            max_tokens = 1500 if use_expensive else 800
            
            if not self.cost_tracker.can_afford(model_name, total_input_tokens + max_tokens):
                return "⚠️ Insufficient budget for this request. Use `!budget` to check remaining funds."
            
            # Make API call with proper conversation structure and short response mode retry logic
            response_text = await self._generate_response_with_conversation_retries(
                model_config, system_message, conversation_messages, max_tokens, model_name, total_input_tokens, message
            )
            
            # Add assistant response to conversation thread
            self._add_to_conversation_thread(message, 'assistant', response_text)
            
            output_tokens = len(self.tokenizer.encode(response_text))
            cost = self.cost_tracker.calculate_cost(model_name, total_input_tokens, output_tokens)
            
            # Record cost
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model=model_name,
                input_tokens=total_input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                action='chat_response',
                user_id=message.author.id,
                channel_id=message.channel.id
            ))
            
            # Store interaction in memory
            interaction_content = f"User: {content}\nBot: {response_text}"
            self.memory_system.store_memory(
                interaction_content,
                message.author.id,
                message.channel.id,
                memory_type='interaction',
                tags=['chat', 'conversation']
            )
            
            # Check for and execute function calls
            function_calls = self.parse_function_calls(response_text)
            if function_calls:
                logger.info(f"🔧 Executing {len(function_calls)} function call(s): {[call['function'] for call in function_calls]}")
                
                # Check if there are any message_user calls
                message_user_calls = [call for call in function_calls if call.get('function') == 'message_user']
                
                if message_user_calls:
                    # Handle message_user calls specially - use their tagged_message directly
                    logger.info(f"🏷️ Found {len(message_user_calls)} message_user calls, handling specially")
                    
                    # Execute the function calls to get the results
                    function_results = await self.execute_function_calls(function_calls)
                    
                    # Parse the function results to find message_user results
                    import json
                    result_lines = function_results.split('\n\n')
                    
                    for line in result_lines:
                        if 'message_user' in line and 'tagged_message' in line:
                            try:
                                # Extract the JSON result
                                json_start = line.find('{')
                                if json_start != -1:
                                    json_part = line[json_start:]
                                    result_data = json.loads(json_part)
                                    
                                    if 'tagged_message' in result_data:
                                        tagged_message = result_data['tagged_message']
                                        logger.info(f"🏷️ Using tagged message: {tagged_message[:100]}...")
                                        
                                        # Update conversation thread with the tagged message
                                        self._add_to_conversation_thread(message, 'assistant', tagged_message)
                                        
                                        # Apply short response mode and return the tagged message
                                        final_message = self._enforce_short_response_mode(tagged_message)
                                        return final_message
                            except:
                                continue
                    
                    # Fallback if we couldn't parse the tagged message
                    logger.warning("⚠️ Could not extract tagged_message from message_user result")
                
                # Execute function calls and get results (for non-message_user functions)
                function_results = await self.execute_function_calls(function_calls)
                logger.info(f"✅ Function calls completed, generating response with results...")
                
                # Create a new prompt that includes the function results
                function_context = f"""
## Function Call Results

The AI assistant just executed these function calls:
{function_results}

Please provide a natural response to the user that incorporates these function results. Be conversational and helpful, presenting the information in a user-friendly way. Don't mention the technical function call process - just naturally integrate the results into your response.

Original user message: {content}
Original AI reasoning: {response_text}
"""
                
                # Generate a new response incorporating the function results with short response mode retry logic
                new_input_tokens = len(self.tokenizer.encode(function_context))
                new_max_tokens = 800 if use_expensive else 500
                
                if self.cost_tracker.can_afford(model_name, new_input_tokens + new_max_tokens):
                    final_response = await self._generate_response_with_short_mode_retries(model_config, function_context, new_max_tokens, model_name, new_input_tokens, message)
                    new_output_tokens = len(self.tokenizer.encode(final_response))
                    new_cost = self.cost_tracker.calculate_cost(model_name, new_input_tokens, new_output_tokens)
                    
                    # Record the additional cost
                    self.cost_tracker.record_usage(CostEntry(
                        timestamp=datetime.now(),
                        model=model_name,
                        input_tokens=new_input_tokens,
                        output_tokens=new_output_tokens,
                        cost=new_cost,
                        action='function_response',
                        user_id=message.author.id,
                        channel_id=message.channel.id
                    ))
                    
                    # Process any user tags from function results (deduplicate)
                    user_tags = set()  # Use set to prevent duplicates
                    tag_calls = 0
                    for call in function_calls:
                        if call.get('function') == 'tag_user':
                            tag_calls += 1
                            user_id = call.get('parameters', {}).get('user_id')
                            if user_id:
                                user_tags.add(f"<@{user_id}>")
                    
                    # Log tagging behavior for debugging
                    if tag_calls > 0:
                        logger.info(f"🏷️ Processed {tag_calls} tag_user calls → {len(user_tags)} unique user tags")
                    
                    # Add user tags to response if any (convert back to list for joining)
                    if user_tags:
                        final_response = f"{' '.join(list(user_tags))} {final_response}".strip()
                    
                    # Update conversation thread with the final response
                    self._add_to_conversation_thread(message, 'assistant', final_response)
                    
                    # Clean any remaining XML tags from the final response
                    cleaned_response = self._clean_xml_tags(final_response)
                    # Apply short response mode limits
                    final_response = self._enforce_short_response_mode(cleaned_response)
                    return final_response
                else:
                    # If we can't afford the follow-up, return a budget warning
                    return "⚠️ Function executed but insufficient budget for detailed response. Use `!budget` to check remaining funds."
            
            # Clean any XML tags from response text when no function calls
            cleaned_response = self._clean_xml_tags(response_text)
            # Apply short response mode limits
            final_response = self._enforce_short_response_mode(cleaned_response)
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, encountered an error: {str(e)[:100]}..."

    async def _call_model_api(self, model_config: Dict, prompt: str, max_tokens: int = 800) -> str:
        """Call the appropriate API based on model configuration."""
        provider = model_config.get('provider', 'anthropic')
        model = model_config.get('model', 'claude-3-haiku-20240307')
        
        if provider == 'anthropic':
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.content[0].text
            
        elif provider == 'openrouter':
            if not hasattr(self, 'openrouter_client'):
                raise Exception("OpenRouter client not initialized - check OPEN_ROUTER_KEY")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openrouter_client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.choices[0].message.content
            
        elif provider == 'openai':
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.choices[0].message.content
            
        else:
            raise Exception(f"Unsupported provider: {provider}")

    async def _call_model_api_with_conversation(self, model_config: Dict, system_message: str, 
                                               conversation_messages: List[Dict[str, str]], 
                                               max_tokens: int = 800) -> str:
        """Call the appropriate API with proper conversation structure."""
        provider = model_config.get('provider', 'anthropic')
        model = model_config.get('model', 'claude-3-haiku-20240307')
        
        if provider == 'anthropic':
            # Anthropic uses system parameter + messages
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_message,
                    messages=conversation_messages
                )
            )
            return response.content[0].text
            
        elif provider == 'openrouter':
            if not hasattr(self, 'openrouter_client'):
                raise Exception("OpenRouter client not initialized - check OPEN_ROUTER_KEY")
            
            # OpenRouter uses system message as first message
            messages = [{"role": "system", "content": system_message}] + conversation_messages
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openrouter_client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages
                )
            )
            return response.choices[0].message.content
            
        elif provider == 'openai':
            # OpenAI uses system message as first message
            messages = [{"role": "system", "content": system_message}] + conversation_messages
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages
                )
            )
            return response.choices[0].message.content
            
        else:
            raise Exception(f"Unsupported provider: {provider}")

    async def _generate_response_with_conversation_retries(self, model_config, system_message, conversation_messages, max_tokens, model_name, input_tokens, message):
        """Generate response with conversation structure and retry logic for short response mode."""
        short_response_mode = self.config.get('bot.short_response_mode', False)
        short_response_limit = self.config.get('bot.short_response_limit', 240)
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Response too long, retrying (attempt {attempt + 1}/{max_retries + 1})")
                
                # Add length constraint message for retries
                current_system_message = system_message
                if attempt > 0 and short_response_mode:
                    current_system_message = system_message + f"""

CRITICAL: Your previous response was too long. You MUST provide a response that is {short_response_limit} characters or less. Be extremely concise. Function results are internal processing only - DO NOT include them in your response text."""
                
                # Generate response
                response = await self._call_model_api_with_conversation(
                    model_config, current_system_message, conversation_messages, max_tokens
                )
                
                # Check length if short mode is enabled (check AFTER cleaning XML tags)
                if short_response_mode:
                    cleaned_for_check = self._clean_xml_tags(response)
                    if len(cleaned_for_check) > short_response_limit:
                        if attempt < max_retries:
                            logger.warning(f"⚠️ Response too long ({len(cleaned_for_check)} chars > {short_response_limit} limit after cleaning), retrying...")
                            continue
                        else:
                            # Final attempt - send full response instead of truncating
                            logger.warning(f"⚠️ Response still too long after {max_retries} retries ({len(cleaned_for_check)} chars), sending full response")
                            response = cleaned_for_check
                    elif len(cleaned_for_check) == 0:
                        if attempt < max_retries:
                            logger.warning(f"⚠️ Response became empty after cleaning XML tags, retrying...")
                            continue
                        else:
                            logger.error(f"❌ Response still empty after {max_retries} retries")
                            response = "Hello! 👋"  # Fallback response
                    else:
                        logger.info(f"✅ Response length OK: {len(cleaned_for_check)}/{short_response_limit} characters")
                else:
                    logger.info(f"✅ Response generated: {len(response)} characters")
                
                return response
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"⚠️ Error in response generation attempt {attempt + 1}: {e}")
                    continue
                else:
                    raise e

    async def _generate_response_with_short_mode_retries(self, model_config, prompt, max_tokens, model_name, input_tokens, message):
        """Generate response with retry logic for short response mode."""
        short_response_mode = self.config.get('bot.short_response_mode', False)
        short_response_limit = self.config.get('bot.short_response_limit', 240)
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Response too long, retrying (attempt {attempt + 1}/{max_retries + 1})")
                
                # Add length constraint message for retries
                current_prompt = prompt
                if attempt > 0 and short_response_mode:
                    current_prompt = prompt + f"""

CRITICAL: Your previous response was too long. You MUST provide a response that is {short_response_limit} characters or less. Be extremely concise. Function results are internal processing only - DO NOT include them in your response text."""
                
                # Generate response
                response = await self._call_model_api(model_config, current_prompt, max_tokens)
                
                # Check length if short mode is enabled (check AFTER cleaning XML tags)
                if short_response_mode:
                    cleaned_for_check = self._clean_xml_tags(response)
                    if len(cleaned_for_check) > short_response_limit:
                        if attempt < max_retries:
                            logger.warning(f"⚠️ Response too long ({len(cleaned_for_check)} chars > {short_response_limit} limit after cleaning), retrying...")
                            continue
                        else:
                            # Final attempt - send full response instead of truncating
                            logger.warning(f"⚠️ Response still too long after {max_retries} retries ({len(cleaned_for_check)} chars), sending full response")
                            response = cleaned_for_check
                    elif len(cleaned_for_check) == 0:
                        if attempt < max_retries:
                            logger.warning(f"⚠️ Response became empty after cleaning XML tags, retrying...")
                            continue
                        else:
                            logger.error(f"❌ Response still empty after {max_retries} retries")
                            response = "Hello! 👋"  # Fallback response
                    else:
                        logger.info(f"✅ Response length OK: {len(cleaned_for_check)}/{short_response_limit} characters")
                else:
                    logger.info(f"✅ Response generated: {len(response)} characters")
                
                return response
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"⚠️ Error in response generation attempt {attempt + 1}: {e}")
                    continue
                else:
                    raise e

    def _enforce_short_response_mode(self, response: str) -> str:
        """Enforce short response mode limits if enabled."""
        short_response_mode = self.config.get('bot.short_response_mode', False)
        
        if not short_response_mode:
            return response
            
        short_response_limit = self.config.get('bot.short_response_limit', 240)
        
        if len(response) <= short_response_limit:
            logger.info(f"✅ Response length OK: {len(response)}/{short_response_limit} characters")
            return response
        
        # Truncate if over limit
        truncated = response[:short_response_limit-3] + "..."
        logger.warning(f"⚠️ Response truncated: {len(response)} chars → {len(truncated)} chars (limit: {short_response_limit})")
        return truncated

    def _clean_xml_tags(self, text: str) -> str:
        """Remove all XML function call tags from response text."""
        # Remove ALL XML command tags from response
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'<query_memory>.*?</query_memory>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<get_user_history.*?>.*?</get_user_history>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<store_observation.*?>.*?</store_observation>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<update_constitution.*?>.*?</update_constitution>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<get_constitution.*?>.*?</get_constitution>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<tag_user.*?>.*?</tag_user>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<message_user.*?>.*?</message_user>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<google_search>.*?</google_search>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<run_command.*?>.*?</run_command>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<generate_image.*?>.*?</generate_image>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<update_importance_criteria.*?>.*?</update_importance_criteria>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<get_budget.*?>.*?</get_budget>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<get_current_time.*?>.*?</get_current_time>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<send_message.*?>.*?</send_message>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<stop_autonomous_mode.*?>.*?</stop_autonomous_mode>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<summarize_context.*?>.*?</summarize_context>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<edit_memory.*?>.*?</edit_memory>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<delete_memory.*?>.*?</delete_memory>', '', cleaned, flags=re.DOTALL)
        
        # Clean up any extra whitespace/newlines
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Remove multiple blank lines
        cleaned = cleaned.strip()
        
        # Debug: Log if content becomes empty after cleaning
        if not cleaned and text:
            logger.warning(f"⚠️ Response became empty after cleaning XML tags. Original: '{text[:200]}...'")
        
        return cleaned

    @tasks.loop(hours=1)
    async def update_member_cache(self):
        """Update cached member information."""
        if not self.target_guild_id:
            return
            
        guild = self.get_guild(self.target_guild_id)
        if not guild:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                for member in guild.members:
                    if not member.bot:
                        roles = json.dumps([role.name for role in member.roles])
                        conn.execute('''
                            INSERT OR REPLACE INTO member_cache
                            (user_id, username, display_name, roles, joined_at, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            member.id,
                            member.name,
                            member.display_name,
                            roles,
                            member.joined_at.isoformat() if member.joined_at else None,
                            datetime.now().isoformat()
                        ))
            logger.info(f"Updated member cache for {len(guild.members)} members")
        except Exception as e:
            logger.error(f"Error updating member cache: {e}")

    @tasks.loop(hours=6)
    async def budget_monitor(self):
        """Monitor budget and send alerts."""
        remaining = self.cost_tracker.get_remaining_budget()
        total_spent = self.cost_tracker.get_total_spending()
        percentage_used = (total_spent / self.budget) * 100
        
        # Alert thresholds
        if percentage_used >= 90:
            message = f"🚨 **BUDGET ALERT**: {percentage_used:.1f}% used (${remaining:.2f} remaining)"
        elif percentage_used >= 75:
            message = f"⚠️ **Budget Warning**: {percentage_used:.1f}% used (${remaining:.2f} remaining)"
        elif percentage_used >= 50:
            message = f"📊 Budget Update: {percentage_used:.1f}% used (${remaining:.2f} remaining)"
        else:
            return  # No alert needed
            
        # Send to admin channel or log
        logger.warning(message)
    
    @tasks.loop(hours=12)
    async def autonomous_investigation(self):
        """Run autonomous investigation twice daily."""
        try:
            # Check if we have sufficient budget for autonomous investigation
            remaining_budget = self.cost_tracker.get_remaining_budget()
            budget_requirement = self.config.get('autonomous.budget_requirement', 1.0)
            
            if remaining_budget < budget_requirement:
                logger.warning(f"Skipping autonomous investigation: insufficient budget (${remaining_budget:.2f} < ${budget_requirement:.2f})")
                return
            
            if self.autonomous_mode_active:
                logger.info("Skipping autonomous investigation: already in autonomous mode")
                return
            
            # Log the investigation start
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"🔍 Starting scheduled autonomous investigation at {current_time}")
            
            # Trigger autonomous thinking mode
            await self.autonomous_thinking()
            
            logger.info("✅ Autonomous investigation completed")
            
        except Exception as e:
            logger.error(f"Error during autonomous investigation: {e}")
            import traceback
            traceback.print_exc()

    async def autonomous_thinking(self):
        """Autonomous thinking and action mode - triggered by command or AI decision."""
        # Skip if budget is too low or already in autonomous mode
        budget_requirement = self.config.get('autonomous.budget_requirement', 1.0)
        if (self.cost_tracker.get_remaining_budget() < budget_requirement or 
            self.autonomous_mode_active):
            return
        
        logger.info("🤖 Starting autonomous thinking session...")
        
        # Activate autonomous mode
        self.autonomous_mode_active = True
        self.autonomous_tool_calls_used = 0
        self._pending_autonomous_message = None
        
        try:
            # Get current context
            current_context = await self._gather_autonomous_context()
            
            # Generate autonomous thinking prompt with full system context
            base_autonomous_prompt = self.config.get('system_prompt.autonomous_prompt', '')
            
            # Handle both string and array formats for backward compatibility
            if isinstance(base_autonomous_prompt, list):
                base_autonomous_prompt = '\n'.join(base_autonomous_prompt)
            
            thinking_prompt = f"""{self.get_system_prompt()}

{base_autonomous_prompt}

**CURRENT INTELLIGENCE:**
{current_context}

**AVAILABLE TOOLS:**
- <think>strategic analysis</think> - Internal reasoning (hidden from users)
- <query_memory>search terms</query_memory> - Search your intelligence database
- <get_user_history user_id="123">5</get_user_history> - Profile individuals
- <send_message channel_id="123456" channel_name="general">message content</send_message> - Send strategic communications
- <tag_user user_id="123">context</tag_user> - Mention specific people strategically
- <store_observation user_id="123" channel_id="456" tags="role_mapping,project_intel">observation</store_observation> - Store strategic intelligence
- <get_constitution></get_constitution> - Check current mission parameters
- <update_constitution reason="learned something new">new constitution</update_constitution> - Update strategic priorities
- <generate_image quality="medium" size="1024x1024">strategic visual content</generate_image> - Create compelling visuals
- <google_search>relevant strategic research</google_search> - Gather external intelligence
- <get_budget></get_budget> - Check available resources
- <get_current_time></get_current_time> - Time-based strategic context
- <stop_autonomous_mode reason="completed full strategic session">summary</stop_autonomous_mode> - ONLY after 15+ tool calls

**SPECIFIC ACTION SEQUENCES:**
Execute these in order:
1. <think>strategic analysis of current gaps</think>
2. <query_memory>social media manager role responsibilities</query_memory>
3. <query_memory>project completion status member progress</query_memory>
4. <google_search>viral AI research content 2024 social media strategies</google_search>
5. Send strategic messages to active channels (check current context for channel IDs)
6. Continue with more targeted actions based on what you discover..."""

            # Use expensive model for autonomous thinking
            response = await self._generate_autonomous_response(thinking_prompt)
            
            # Process any function calls from the autonomous response
            if response:
                await self._process_autonomous_response(response)
                
        except Exception as e:
            logger.error(f"Error in autonomous thinking: {e}")
        finally:
            # Ensure autonomous mode is deactivated
            if self.autonomous_mode_active:
                self.autonomous_mode_active = False
                logger.info(f"🤖 Autonomous session ended. Used {self.autonomous_tool_calls_used} tool calls.")

    async def _gather_autonomous_context(self) -> str:
        """Gather current context for autonomous decision making."""
        context_parts = []
        
        # Get budget status
        budget_info = self._function_get_budget()
        context_parts.append(f"Budget: {budget_info.get('message', 'Unknown')}")
        
        # Get time status  
        time_info = self._function_get_current_time()
        context_parts.append(f"Time: {time_info.get('message', 'Unknown')}")
        
        # Get recent channel activity and identify key channels
        if self.target_guild_id:
            guild = self.get_guild(self.target_guild_id)
            if guild:
                recent_activity = []
                channel_insights = []
                
                for channel in guild.text_channels[:5]:  # Check top 5 channels
                    try:
                        history = self.data_store.get_channel_history(channel.id, limit=10)
                        if history:
                            recent_activity.append(f"#{channel.name}: {len(history)} recent messages")
                            
                            # Look for strategic keywords in recent messages
                            messages_text = " ".join([msg.get('content', '') for msg in history])
                            strategic_keywords = ['project', 'presentation', 'social media', 'twitter', 'x.com', 'instagram', 'tiktok', 'content', 'marketing', 'campaign', 'deadline', 'completion', 'progress', 'collaboration', 'manager', 'lead', 'responsible']
                            
                            found_keywords = [kw for kw in strategic_keywords if kw.lower() in messages_text.lower()]
                            if found_keywords:
                                channel_insights.append(f"#{channel.name}: mentions {', '.join(found_keywords)}")
                    except:
                        pass
                
                if recent_activity:
                    context_parts.append(f"Channel Activity: {', '.join(recent_activity)}")
                if channel_insights:
                    context_parts.append(f"Strategic Intelligence: {'; '.join(channel_insights)}")
                
                # Add channel IDs for messaging
                active_channels = []
                for channel in guild.text_channels[:3]:  # Top 3 most relevant channels
                    if channel.permissions_for(guild.me).send_messages:
                        active_channels.append(f"#{channel.name} (ID: {channel.id})")
                if active_channels:
                    context_parts.append(f"Available Channels for Messaging: {', '.join(active_channels)}")
        
        # Get member role analysis and identify knowledge gaps
        try:
            role_intelligence = self.memory_system.query_memories(
                "role responsibility manager lead social media project", 
                limit=5
            )
            known_roles = set()
            if role_intelligence:
                roles_found = []
                for obs in role_intelligence:
                    content_lower = obs['content'].lower()
                    if any(keyword in content_lower for keyword in ['manager', 'lead', 'responsible', 'handles', 'in charge']):
                        roles_found.append(obs['content'][:80] + "...")
                        # Track what roles we know about
                        if 'social media' in content_lower:
                            known_roles.add('social_media')
                        if 'project' in content_lower:
                            known_roles.add('project_management')
                        if 'content' in content_lower:
                            known_roles.add('content_creation')
                if roles_found:
                    context_parts.append(f"Known Roles: {'; '.join(roles_found)}")
            
            # Identify critical knowledge gaps
            critical_roles = ['social_media', 'project_management', 'content_creation']
            missing_roles = [role for role in critical_roles if role not in known_roles]
            if missing_roles:
                context_parts.append(f"Knowledge Gaps: Missing intel on {', '.join(missing_roles)} roles - strategic outreach needed")
        except:
            pass
        
        # Get project status intelligence and identify completion risks
        try:
            project_intel = self.memory_system.query_memories(
                "project status completion progress deadline", 
                limit=5
            )
            if project_intel:
                project_status = []
                completion_concerns = []
                for obs in project_intel:
                    content_lower = obs['content'].lower()
                    if any(keyword in content_lower for keyword in ['project', 'progress', 'completion', 'deadline']):
                        project_status.append(obs['content'][:80] + "...")
                        # Look for completion concerns
                        if any(concern in content_lower for concern in ['behind', 'delayed', 'stuck', 'struggling', 'help']):
                            completion_concerns.append(obs['content'][:60] + "...")
                if project_status:
                    context_parts.append(f"Project Intelligence: {'; '.join(project_status)}")
                if completion_concerns:
                    context_parts.append(f"Completion Concerns: {'; '.join(completion_concerns)} - proactive support needed")
        except:
            pass
        
        # Get active members who could be strategic contacts
        if self.target_guild_id:
            guild = self.get_guild(self.target_guild_id)
            if guild:
                try:
                    recent_users = []
                    for channel in guild.text_channels[:3]:
                        history = self.data_store.get_channel_history(channel.id, limit=5)
                        for msg in history:
                            if msg.get('user_id') and msg.get('user_id') not in recent_users:
                                recent_users.append(msg.get('user_id'))
                    if recent_users:
                        context_parts.append(f"Active Members: {len(recent_users)} recently active users available for strategic outreach")
                except:
                    pass
        
        # Check for strategic timing opportunities
        try:
            time_info = self._function_get_current_time()
            if 'days until' in time_info.get('message', ''):
                context_parts.append(f"Strategic Timing: {time_info['message']} - urgency factor for proactive actions")
        except:
            pass
        
        return "\n".join(context_parts)

    async def _generate_autonomous_response(self, prompt: str) -> str:
        """Generate response for autonomous mode using expensive model."""
        try:
            input_tokens = len(self.tokenizer.encode(prompt))
            
            if not self.cost_tracker.can_afford('sonnet', input_tokens + 1000):
                logger.warning("Cannot afford autonomous thinking - skipping")
                return ""
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model=self.expensive_model_config.get('model', 'claude-3-5-sonnet-20241022'),
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            
            output_tokens = len(self.tokenizer.encode(response.content[0].text))
            cost = self.cost_tracker.calculate_cost('sonnet', input_tokens, output_tokens)
            
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model='sonnet',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                action='autonomous_thinking',
                user_id=0,
                channel_id=0
            ))
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error generating autonomous response: {e}")
            return ""

    async def _process_autonomous_response(self, response: str):
        """Process autonomous response and execute tool calls."""
        # Parse function calls from response
        function_calls = self.parse_function_calls(response)
        
        if not function_calls:
            logger.info("No function calls found in autonomous response")
            return
        
        # Execute tool calls with limit
        for call in function_calls:
            if self.autonomous_tool_calls_used >= 25:
                logger.warning("Reached 25 tool call limit in autonomous mode")
                break
                
            if not self.autonomous_mode_active:
                logger.info("Autonomous mode stopped by function call")
                break
            
            # Check budget before each call
            if self.cost_tracker.get_remaining_budget() <= 1.0:
                logger.warning("Low budget - stopping autonomous mode")
                break
                
            # Execute function call
            func_name = call.get('function')
            params = call.get('parameters', {})
            
            try:
                if func_name in self.available_functions:
                    result = self.available_functions[func_name](**params)
                    self.autonomous_tool_calls_used += 1
                    
                    # Store result in context history
                    self.autonomous_context_history.append({
                        'type': 'tool_result',
                        'function': func_name,
                        'parameters': params,
                        'result': result,
                        'success': result.get('success', True),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Handle pending message sends
                    if hasattr(self, '_pending_autonomous_message') and self._pending_autonomous_message:
                        await self._send_pending_autonomous_message()
                    
                    logger.info(f"Autonomous tool call {self.autonomous_tool_calls_used}: {func_name}")
                    
                    # Brief pause between tool calls
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error executing autonomous function {func_name}: {e}")
                self.autonomous_tool_calls_used += 1

    async def _send_pending_autonomous_message(self):
        """Send any pending autonomous messages."""
        if not hasattr(self, '_pending_autonomous_message') or not self._pending_autonomous_message:
            return
            
        try:
            pending = self._pending_autonomous_message
            channel = pending['channel']
            message = pending['message']
            
            await channel.send(message)
            logger.info(f"Sent autonomous message to #{channel.name if hasattr(channel, 'name') else 'DM'}")
            
            # Store the sent message
            self.memory_system.store_memory(
                f"Sent autonomous message to #{channel.name if hasattr(channel, 'name') else 'DM'}: {message[:100]}...",
                user_id=0,
                channel_id=channel.id,
                memory_type='observation',
                tags=['autonomous', 'message_sent']
            )
            
        except Exception as e:
            logger.error(f"Error sending autonomous message: {e}")
        finally:
            self._pending_autonomous_message = None



    def _should_monitor_channel(self, channel) -> bool:
        """Check if a channel should be monitored for messages."""
        # Always monitor DMs
        if isinstance(channel, discord.DMChannel):
            return True
        
        # Monitor channels in target guild (Aug Lab)
        if (hasattr(channel, 'guild') and channel.guild and 
            self.target_guild_id and channel.guild.id == self.target_guild_id):
            return True
        
        # If no target guild set, monitor all guilds (for initial setup)
        if hasattr(channel, 'guild') and channel.guild and not self.target_guild_id:
            return True
            
        return False

    async def update_channel_tracking(self):
        """Update channel tracking status for monitored channels."""
        try:
            # Update tracking for all accessible channels
            for guild in self.guilds:
                if self.target_guild_id and guild.id != self.target_guild_id:
                    continue  # Skip non-target guilds if restricted
                
                for channel in guild.text_channels:
                    if self._should_monitor_channel(channel):
                        # Track monitored channels
                        try:
                            logger.debug(f"Monitoring channel #{channel.name}")
                        except Exception as e:
                            logger.error(f"Error tracking channel #{channel.name}: {e}")
                            
        except Exception as e:
            logger.error(f"Error updating channel tracking: {e}")

    # Commands
    @commands.command(name='budget')
    async def budget_command(self, ctx):
        """Show budget status and spending report."""
        report = self.cost_tracker.get_spending_report()
        
        embed = discord.Embed(
            title="💰 Compute Budget Report",
            color=discord.Color.blue()
        )
        
        percentage_used = (report['total_spent'] / report['budget']) * 100
        embed.add_field(
            name="Budget Status",
            value=f"**Total Budget**: ${report['budget']:.2f}\n"
                  f"**Spent**: ${report['total_spent']:.2f} ({percentage_used:.1f}%)\n"
                  f"**Remaining**: ${report['remaining']:.2f}",
            inline=False
        )
        
        if report['by_model']:
            model_info = []
            for model, data in report['by_model'].items():
                model_info.append(f"**{model.title()}**: ${data['cost']:.2f} ({data['requests']} requests)")
            embed.add_field(name="By Model", value="\n".join(model_info), inline=False)
        
        await ctx.send(embed=embed)

    @commands.command(name='constitution')
    async def constitution_command(self, ctx):
        """Show the current constitution."""
        constitution = self.constitution_manager.get_current_constitution()
        
        if len(constitution) > 1900:  # Discord limit
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

    @commands.command(name='memory_query')
    async def memory_query_command(self, ctx, *, query):
        """Query the bot's long-term memory."""
        memories = self.memory_system.query_memories(query, limit=5)
        
        if not memories:
            await ctx.send(f"🔍 No memories found for query: '{query}'")
            return
        
        embed = discord.Embed(
            title=f"🧠 Memory Search: '{query}'",
            description=f"Found {len(memories)} relevant memories",
            color=discord.Color.green()
        )
        
        for i, memory in enumerate(memories[:3], 1):  # Show top 3
            timestamp = memory['timestamp'][:19]
            content = memory['content'][:200] + "..." if len(memory['content']) > 200 else memory['content']
            
            # Add similarity score if available
            similarity_info = ""
            if 'similarity' in memory and memory['similarity'] > 0:
                similarity_info = f" (similarity: {memory['similarity']:.2f})"
                
            embed.add_field(
                name=f"Memory {i} - {memory['memory_type'].title()} ({timestamp}){similarity_info}",
                value=content,
                inline=False
            )
        
        await ctx.send(embed=embed)

    @commands.command(name='user_history')
    async def user_history_command(self, ctx, user: discord.Member = None):
        """Show interaction history for a user (defaults to yourself)."""
        target_user = user or ctx.author
        
        # Only allow checking own history or if user has manage messages permission
        if target_user != ctx.author and not ctx.author.guild_permissions.manage_messages:
            await ctx.send("❌ You can only check your own history unless you have manage messages permission.")
            return
            
        history = self.memory_system.get_user_interaction_history(target_user.id, 5)
        
        if not history:
            await ctx.send(f"📊 No interaction history found for {target_user.display_name}")
            return
        
        embed = discord.Embed(
            title=f"📊 Interaction History: {target_user.display_name}",
            description=f"Last {len(history)} interactions",
            color=discord.Color.purple()
        )
        
        for i, interaction in enumerate(history, 1):
            timestamp = interaction['timestamp'][:19]
            content = interaction['content'][:150] + "..." if len(interaction['content']) > 150 else interaction['content']
            embed.add_field(
                name=f"Interaction {i} ({timestamp})",
                value=content,
                inline=False
            )
        
        await ctx.send(embed=embed)

    @commands.command(name='memory_stats')
    async def memory_stats_command(self, ctx):
        """Show memory system statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count memories by type
                cursor = conn.execute('''
                    SELECT memory_type, COUNT(*) 
                    FROM memories 
                    GROUP BY memory_type
                ''')
                memory_counts = dict(cursor.fetchall())
                
                # Total memories
                cursor = conn.execute('SELECT COUNT(*) FROM memories')
                total_memories = cursor.fetchone()[0]
                
                # Total connections
                cursor = conn.execute('SELECT COUNT(*) FROM memory_connections')
                total_connections = cursor.fetchone()[0]
                
                # Recent activity
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM memories 
                    WHERE datetime(timestamp) > datetime('now', '-24 hours')
                ''')
                recent_memories = cursor.fetchone()[0]
            
            embed = discord.Embed(
                title="🧠 Memory System Statistics",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="Total Memories",
                value=f"**{total_memories:,}** stored memories",
                inline=True
            )
            
            embed.add_field(
                name="Memory Connections",
                value=f"**{total_connections:,}** relationships",
                inline=True
            )
            
            embed.add_field(
                name="Recent Activity",
                value=f"**{recent_memories}** memories in last 24h",
                inline=True
            )
            
            if memory_counts:
                types_text = "\n".join([f"• {mtype.title()}: {count}" for mtype, count in memory_counts.items()])
                embed.add_field(
                    name="Memory Types",
                    value=types_text,
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error retrieving memory stats: {str(e)}")

    @commands.command(name='dm_residency')
    async def dm_residency_command(self, ctx, *, message_content):
        """DM all members of the 25 residency group."""
        if not ctx.guild or ctx.guild.id != self.target_guild_id:
            await ctx.send("❌ This command can only be used in the target guild.")
            return
            
        residency_role = discord.utils.get(ctx.guild.roles, name=self.residency_role_name)
        if not residency_role:
            await ctx.send(f"❌ Role '{self.residency_role_name}' not found.")
            return
            
        members = [member for member in residency_role.members if not member.bot]
        success_count = 0
        
        embed = discord.Embed(
            title="Message from Augmentation Lab",
            description=message_content,
            color=discord.Color.purple(),
            timestamp=datetime.now()
        )
        embed.set_footer(text=f"Sent by {ctx.author.display_name}")
        
        for member in members:
            try:
                await member.send(embed=embed)
                success_count += 1
                await asyncio.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Could not DM {member.display_name}: {e}")
        
        await ctx.send(f"✅ Successfully sent DM to {success_count}/{len(members)} residency members.")

    @commands.command(name='search')
    async def search_command(self, ctx, *, query):
        """Search stored channel data."""
        results = self.data_store.search_content(query, ctx.channel.id if ctx.guild else None)
        
        if not results:
            await ctx.send("🔍 No results found.")
            return
            
        embed = discord.Embed(title=f"🔍 Search Results for '{query}'", color=discord.Color.green())
        
        for i, result in enumerate(results[:5]):  # Limit to 5 results
            embed.add_field(
                name=f"{result['channel']} - {result['author']}",
                value=f"{result['content'][:100]}...\n*{result['timestamp'][:19]}*",
                inline=False
            )
        
        if len(results) > 5:
            embed.set_footer(text=f"Showing 5 of {len(results)} results")
            
        await ctx.send(embed=embed)

    @commands.command(name='generate_image')
    async def generate_image_command(self, ctx, *, prompt):
        """Generate an image using GPT-4o's native image generation."""
        # Check budget for medium quality image (~$0.04)
        if not self.cost_tracker.can_afford('gpt4o', 1200):  # ~1000 output tokens + 200 input
            await ctx.send("⚠️ Insufficient budget for image generation.")
            return
            
        try:
            # Parse quality and size from prompt (optional)
            quality = "medium"  # Default
            size = "1024x1024"  # Default
            
            # Simple parsing for quality/size overrides
            if "high quality" in prompt.lower() or "hd" in prompt.lower():
                quality = "high"
            elif "low quality" in prompt.lower() or "draft" in prompt.lower():
                quality = "low"
                
            async with ctx.typing():
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.openai_client.images.generate(
                        model="gpt-image-1",
                        prompt=prompt,
                        size=size,
                        quality=quality,
                        n=1
                    )
                )
            
            # Calculate accurate cost based on quality
            input_tokens = len(prompt.split()) * 1.25  # More accurate estimate
            if quality == "low":
                output_tokens = 272  # ~$0.011
                actual_cost = 0.011
            elif quality == "high":
                output_tokens = 4160  # ~$0.166
                actual_cost = 0.166
            else:  # medium
                output_tokens = 1056  # ~$0.042
                actual_cost = 0.042
                
            # Add text input cost
            actual_cost += (input_tokens / 1000000) * 5.0  # $5.00 per 1M tokens
            
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model='gpt-image-1',
                input_tokens=int(input_tokens),
                output_tokens=output_tokens,
                cost=actual_cost,
                action='image_generation',
                user_id=ctx.author.id,
                channel_id=ctx.channel.id
            ))
            
            embed = discord.Embed(
                title="🎨 Generated Image (GPT-4o)",
                description=f"**Prompt:** {prompt}\n**Quality:** {quality.title()}\n**Cost:** ${actual_cost:.3f}",
                color=discord.Color.purple()
            )
            embed.set_image(url=response.data[0].url)
            embed.set_footer(text=f"Generated for {ctx.author.display_name} | Budget: ${self.cost_tracker.get_remaining_budget():.2f}")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error generating image: {str(e)[:100]}...")

    @commands.command(name='help')
    async def help_command(self, ctx):
        """Show help information."""
        embed = discord.Embed(
            title="🧪 Augmentation Lab Bot",
            description="Your intelligent lab assistant with advanced memory and self-evolution",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="🧠 Strategic Monitoring",
            value="• **Haiku** monitors ALL Aug Lab messages\n"
                  "• Decides strategic value & escalates to **Sonnet 4**\n"
                  "• Tracks project progress & social media opportunities\n"
                  "• Smart budget management to last all summer",
            inline=False
        )
        
        embed.add_field(
            name="📊 Budget & Memory Commands",
            value="`!budget` - View spending report\n"
                  "`!constitution` - View current constitution\n"
                  "`!memory_query <text>` - Search memories\n"
                  "`!user_history [@user]` - View interaction history\n"
                  "`!memory_stats` - Memory system statistics",
            inline=False
        )
        
        embed.add_field(
            name="🛠️ Lab Commands",
            value="`!search <query>` - Search channel history\n"
                  "`!generate_image <prompt>` - Create images with GPT-4o\n"
                  "`!dm_residency <message>` - Message residents\n"
                  "`!guilds` - List Discord servers bot is in\n"
                  "`!models` - View available AI models\n"
                  "`!switch_model <cheap|expensive> <model>` - Switch models\n"
                  "`!response_mode [mode]` - Configure response behavior",
            inline=False
        )
        
        embed.add_field(
            name="🤖 Autonomous Mode",
            value="`!autonomous_think` - Trigger autonomous mode (admin)\n"
                  "`!autonomous_status` - Check autonomous mode status\n"
                  "• **Command/AI-triggered** - no automatic scheduling\n"
                  "• **Up to 25 tool calls** - research, message, analyze\n"
                  "• **Budget aware** - automatically manages spending",
            inline=False
        )
        
        embed.add_field(
            name="🤖 How I Work",
            value="• **Always listening** in Aug Lab channels for opportunities\n"
                  "• **Mention me** for direct responses\n"
                  "• **Keywords** like 'project', 'social media', 'viral' trigger monitoring\n"
                  "• **Memory system** learns & connects lab member interactions",
            inline=False
        )
        
        remaining = self.cost_tracker.get_remaining_budget()
        embed.set_footer(text=f"Budget remaining: ${remaining:.2f} | Advanced AI with evolving constitution")
        
        await ctx.send(embed=embed)

    @commands.command(name='autonomous_think')
    async def autonomous_think_command(self, ctx):
        """Manually trigger autonomous thinking mode (admin only)."""
        # Check if user has manage messages permission (basic admin check)
        if not ctx.author.guild_permissions.manage_messages:
            await ctx.send("❌ You need manage messages permission to trigger autonomous mode.")
            return
        
        if self.autonomous_mode_active:
            await ctx.send("🤖 Autonomous mode is already active!")
            return
            
        budget_requirement = self.config.get('autonomous.budget_requirement', 1.0)
        if self.cost_tracker.get_remaining_budget() < budget_requirement:
            await ctx.send(f"❌ Insufficient budget for autonomous mode (${self.cost_tracker.get_remaining_budget():.2f} remaining)")
            return
        
        await ctx.send("🤖 **Triggering autonomous thinking mode...**\nThis may take a few minutes. Check logs for progress.")
        
        # Trigger autonomous thinking manually
        try:
            await self.autonomous_thinking()
            await ctx.send(f"✅ **Autonomous session completed!** Used {self.autonomous_tool_calls_used} tool calls.")
        except Exception as e:
            await ctx.send(f"❌ **Error in autonomous mode:** {str(e)[:200]}")
            logger.error(f"Manual autonomous mode error: {e}")

    @commands.command(name='autonomous_status')
    async def autonomous_status_command(self, ctx):
        """Check autonomous mode status."""
        embed = discord.Embed(
            title="🤖 Autonomous Mode Status",
            color=discord.Color.blue()
        )
        
        if self.autonomous_mode_active:
            embed.add_field(
                name="Status", 
                value=f"🟢 **ACTIVE** (Tool calls used: {self.autonomous_tool_calls_used}/25)",
                inline=False
            )
        else:
            embed.add_field(
                name="Status",
                value="🔴 **INACTIVE**",
                inline=False
            )
        
        # Show trigger methods
        embed.add_field(
            name="Trigger Methods",
            value="• `!autonomous_think` command (admin)\n• AI can trigger via `<trigger_autonomous_mode>` function\n• No automatic scheduling",
            inline=False
        )
        
        # Recent autonomous activity
        recent_autonomous = self.memory_system.query_memories(
            "autonomous", 
            memory_type='observation',
            limit=3
        )
        
        if recent_autonomous:
            recent_activity = []
            for activity in recent_autonomous:
                timestamp = activity['timestamp'][:19]
                content = activity['content'][:100]
                recent_activity.append(f"**{timestamp}:** {content}")
            
            embed.add_field(
                name="Recent Autonomous Activity",
                value="\n".join(recent_activity),
                inline=False
            )
        
        await ctx.send(embed=embed)

    @commands.command(name='models')
    async def models_command(self, ctx):
        """Show available models and current configuration."""
        embed = discord.Embed(
            title="🧠 Model Configuration",
            description="Current model setup and available options",
            color=0x00aaff
        )
        
        # Current models
        cheap_config = self.cheap_model_config
        expensive_config = self.expensive_model_config
        
        embed.add_field(
            name="💰 Cheap Model (Default)",
            value=f"**{cheap_config.get('name', 'haiku')}**\n"
                  f"Provider: {cheap_config.get('provider', 'anthropic')}\n"
                  f"Model: {cheap_config.get('model', 'claude-3-haiku-20240307')}",
            inline=True
        )
        
        embed.add_field(
            name="💎 Expensive Model (Complex tasks)",
            value=f"**{expensive_config.get('name', 'sonnet')}**\n"
                  f"Provider: {expensive_config.get('provider', 'anthropic')}\n"
                  f"Model: {expensive_config.get('model', 'claude-3-5-sonnet-20241022')}",
            inline=True
        )
        
        # Available models
        available_models = self.config.list_available_models()
        if available_models:
            models_text = "\n".join([f"• {model}" for model in available_models])
            embed.add_field(
                name="🎯 Available Models",
                value=models_text,
                inline=False
            )
        
        embed.add_field(
            name="🔧 Usage",
            value="`!switch_model <cheap|expensive> <model_name>`\n"
                  "Example: `!switch_model expensive grok4`",
            inline=False
        )
        
        await ctx.send(embed=embed)

    @commands.command(name='switch_model')
    async def switch_model_command(self, ctx, model_type: str = None, model_name: str = None):
        """Switch between available models."""
        if not model_type or not model_name:
            await ctx.send("❌ Usage: `!switch_model <cheap|expensive> <model_name>`\n"
                          "Use `!models` to see available options.")
            return
        
        model_type = model_type.lower()
        if model_type not in ["cheap", "expensive"]:
            await ctx.send("❌ Model type must be 'cheap' or 'expensive'")
            return
        
        # Check if model exists
        available_models = self.config.list_available_models()
        if model_name not in available_models:
            await ctx.send(f"❌ Model '{model_name}' not found.\n"
                          f"Available models: {', '.join(available_models)}")
            return
        
        # Switch the model
        success = self.config.switch_model(model_type, model_name)
        if success:
            # Update the bot's model config
            self.cheap_model_config = self.config.get_model_config('cheap')
            self.expensive_model_config = self.config.get_model_config('expensive')
            
            model_config = self.config.get_model_config(model_type)
            embed = discord.Embed(
                title="✅ Model Switched Successfully",
                description=f"Updated **{model_type}** model configuration",
                color=0x00ff00
            )
            
            embed.add_field(
                name="New Configuration",
                value=f"**Name:** {model_config.get('name')}\n"
                      f"**Provider:** {model_config.get('provider')}\n"
                      f"**Model:** {model_config.get('model')}",
                inline=False
            )
            
            await ctx.send(embed=embed)
        else:
            await ctx.send("❌ Failed to switch model. Please try again.")

    @commands.command(name='guilds')
    async def guilds_command(self, ctx):
        """List all Discord servers (guilds) the bot is currently in."""
        embed = discord.Embed(
            title="🌐 Discord Servers",
            description="Servers where this bot is currently active",
            color=discord.Color.blue()
        )
        
        if not self.guilds:
            embed.add_field(
                name="No Servers",
                value="Bot is not in any Discord servers",
                inline=False
            )
        else:
            for guild in self.guilds:
                member_count = len([m for m in guild.members if not m.bot])
                bot_permissions = guild.get_member(self.user.id).guild_permissions
                
                permissions_text = "✅ Admin" if bot_permissions.administrator else "🔒 Limited"
                if bot_permissions.manage_messages and bot_permissions.send_messages:
                    permissions_text += " (Can moderate)"
                elif bot_permissions.send_messages:
                    permissions_text += " (Can send messages)"
                
                embed.add_field(
                    name=f"🏠 {guild.name}",
                    value=f"**ID:** {guild.id}\n"
                          f"**Members:** {member_count} humans + {len(guild.members) - member_count} bots\n"
                          f"**Owner:** {guild.owner.display_name if guild.owner else 'Unknown'}\n"
                          f"**Permissions:** {permissions_text}\n"
                          f"**Target Guild:** {'✅ YES' if guild.id == self.target_guild_id else '❌ No'}",
                    inline=False
                )
        
        # Add restriction status
        restriction_status = "🔒 RESTRICTED" if self.restricted_mode else "🔓 OPEN"
        if self.restricted_mode and self.allowed_guilds:
            restriction_status += f" (Allow: {len(self.allowed_guilds)} guilds)"
        elif self.restricted_mode:
            restriction_status += " (Allow: None - bot inactive)"
        
        # Add instructions for removing bot from unwanted servers
        embed.add_field(
            name="📝 Management & Restrictions",
            value=f"**Mode:** {restriction_status}\n"
                  f"**Target Guild:** {self.target_guild_id or 'Not set'}\n"
                  f"**Allowed Guilds:** {', '.join(map(str, self.allowed_guilds)) if self.allowed_guilds else 'All guilds'}\n"
                  "• To restrict bot to specific servers: Edit config.json\n"
                  "• To remove from a server: Use Discord's server settings",
            inline=False
        )
        
        await ctx.send(embed=embed)

    def _should_respond_to_message(self, message) -> bool:
        """Determine if the bot should respond to this message based on response_mode."""
        logger.info(f"🎯 Checking response criteria: mode={self.response_mode}, is_dm={isinstance(message.channel, discord.DMChannel)}, is_mentioned={self.user in message.mentions}")
        
        if self.response_mode == "disabled":
            logger.info("🎯 Response mode is disabled")
            return False
        
        # Always respond to DMs (unless disabled)
        if isinstance(message.channel, discord.DMChannel):
            should_respond = self.response_mode in ["all_messages", "mentions_only", "dms_only"]
            logger.info(f"🎯 DM detected: should_respond={should_respond} (mode allows DMs: {self.response_mode in ['all_messages', 'mentions_only', 'dms_only']})")
            return should_respond
        
        # Always respond to direct mentions (unless disabled or dms_only)
        if self.user in message.mentions:
            should_respond = self.response_mode in ["all_messages", "mentions_only"]
            logger.info(f"🎯 Direct mention detected: should_respond={should_respond} (mode allows mentions: {self.response_mode in ['all_messages', 'mentions_only']})")
            return should_respond
        
        # Respond to all messages in guilds if mode is all_messages
        if self.response_mode == "all_messages":
            result = (
                # Monitor all messages in target guild (Aug Lab) if set
                (message.guild and self.target_guild_id and message.guild.id == self.target_guild_id) or
                # If no target guild set, monitor all guilds (for initial setup)
                (message.guild and not self.target_guild_id) or
                # Monitor if contains Aug Lab keywords
                any(word in message.content.lower() for word in ['auglab', 'lab', 'project', 'presentation', 'social media', 'twitter', 'x.com', 'viral'])
            )
            logger.info(f"🎯 all_messages mode: should_respond={result}")
            return result
        
        logger.info("🎯 No criteria met for response")
        return False

    async def memory_monitor(self, message):
        """Monitor messages for memory storage opportunities without responding."""
        if self.cost_tracker.get_remaining_budget() <= 0:
            return  # Don't monitor if no budget
            
        # Clean message content
        content = message.content
        if self.user in message.mentions:
            content = content.replace(f'<@{self.user.id}>', '').strip()
        if content.startswith(self.prefix):
            content = content[len(self.prefix):].strip()
            
        if not content or len(content) < 3:
            return  # Skip very short messages
            
        try:
            # Use Haiku to assess memory value without responding
            memory_assessment_prompt = f"""You are monitoring messages for the Aug Lab Discord bot to decide if they should be stored in memory. You will NOT respond to the user - this is purely for memory assessment.

## YOUR TASK
Assess if this message contains strategic value for the Augmentation Lab and should be stored in memory.

## MESSAGE TO ASSESS
Message: "{content}"
Author: {message.author.display_name}
Channel: #{getattr(message.channel, 'name', 'DM')}
Context: Aug Lab summer program focused on hitting specific metrics (social media virality, project completion, member satisfaction)

## ASSESSMENT CRITERIA
- Strategic planning or project updates
- Member capabilities, skills, or interests
- Important announcements or decisions
- Collaboration opportunities
- Research insights or technical discussions
- Social connections or networking
- Achievement or milestone mentions

Respond with JSON only:
{{
    "should_store": true/false,
    "importance": 1-5,
    "tags": ["tag1", "tag2", "tag3"],
    "reasoning": "brief explanation"
}}

Be selective - only store truly valuable strategic information."""

            input_tokens = len(self.tokenizer.encode(memory_assessment_prompt))
            
            if not self.cost_tracker.can_afford('haiku', input_tokens + 100):
                return  # Skip if can't afford monitoring
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model=self.cheap_model_config.get('model', 'claude-3-haiku-20240307'),
                    max_tokens=150,
                    messages=[{"role": "user", "content": memory_assessment_prompt}]
                )
            )
            
            output_tokens = len(self.tokenizer.encode(response.content[0].text))
            cost = self.cost_tracker.calculate_cost('haiku', input_tokens, output_tokens)
            
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model='haiku',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                action='memory_monitoring',
                user_id=message.author.id,
                channel_id=message.channel.id
            ))
            
            # Parse Haiku's assessment
            try:
                import json
                assessment = json.loads(response.content[0].text)
                
                # Store in memory if deemed valuable
                if assessment.get('should_store', False):
                    memory_tags = assessment.get('tags', [])
                    memory_tags.extend(['monitoring', 'auto_stored'])
                    
                    self.memory_system.store_memory(
                        f"Auto-stored observation: {content}",
                        message.author.id,
                        message.channel.id,
                        memory_type='observation',
                        tags=memory_tags
                    )
                    
                    logger.info(f"🧠 Auto-stored memory (importance: {assessment.get('importance', 0)}): {content[:50]}...")
                    
            except json.JSONDecodeError:
                logger.warning("Failed to parse memory assessment JSON")
                
        except Exception as e:
            logger.error(f"Error in memory monitoring: {e}")

    @commands.command(name='response_mode')
    async def response_mode_command(self, ctx, mode: str = None):
        """Check or change the bot's response mode."""
        if mode is None:
            # Show current configuration
            embed = discord.Embed(
                title="🤖 Bot Response Configuration",
                description="Current response behavior settings",
                color=0x00aaff
            )
            
            embed.add_field(
                name="Response Mode",
                value=f"**{self.response_mode}**",
                inline=True
            )
            
            embed.add_field(
                name="Memory Monitoring",
                value=f"**{'Enabled' if self.always_monitor_memory else 'Disabled'}**",
                inline=True
            )
            
            embed.add_field(
                name="Available Modes",
                value="• `all_messages` - Respond to all messages\n"
                      "• `mentions_only` - Only respond to @mentions and DMs\n"
                      "• `dms_only` - Only respond to DMs\n"
                      "• `disabled` - Don't respond to anything",
                inline=False
            )
            
            embed.add_field(
                name="Usage",
                value="`!response_mode <mode>` - Change response mode\n"
                      "`!response_mode` - Show current settings",
                inline=False
            )
            
            await ctx.send(embed=embed)
            return
        
        # Validate mode
        valid_modes = ["all_messages", "mentions_only", "dms_only", "disabled"]
        if mode not in valid_modes:
            await ctx.send(f"❌ Invalid mode. Choose from: {', '.join(valid_modes)}")
            return
        
        # Update configuration
        old_mode = self.response_mode
        self.response_mode = mode
        self.config.set('bot.response_mode', mode)
        
        embed = discord.Embed(
            title="✅ Response Mode Updated",
            description=f"Changed from **{old_mode}** to **{mode}**",
            color=0x00ff00
        )
        
        if mode == "all_messages":
            embed.add_field(
                name="Behavior",
                value="Bot will respond to all messages in monitored channels",
                inline=False
            )
        elif mode == "mentions_only":
            embed.add_field(
                name="Behavior", 
                value="Bot will only respond to @mentions and DMs",
                inline=False
            )
        elif mode == "dms_only":
            embed.add_field(
                name="Behavior",
                value="Bot will only respond to direct messages",
                inline=False
            )
        elif mode == "disabled":
            embed.add_field(
                name="Behavior",
                value="Bot will not respond to any messages",
                inline=False
            )
        
        if self.always_monitor_memory:
            embed.add_field(
                name="Memory Monitoring",
                value="✅ Still monitoring all messages for memory storage",
                inline=False
            )
        
        await ctx.send(embed=embed)



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
        
    # OpenAI is optional
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        logger.warning("OPENAI_API_KEY not found - image generation will be disabled")
    
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

if __name__ == "__main__":
    main() 