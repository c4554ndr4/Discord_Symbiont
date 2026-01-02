"""
Memory system for Augmentation Lab Bot - OpenAI Embeddings Version
"""

import json
import sqlite3
import logging
import asyncio
import struct
import threading
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import openai

logger = logging.getLogger(__name__)

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

class MemorySystem:
    """Graph-based long-term memory system with OpenAI embeddings."""
    
    def __init__(self, db_path: str, openai_api_key: str = None):
        self.db_path = db_path
        self.init_memory_tables()
        
        # Initialize OpenAI client for embeddings
        try:
            self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
            self.semantic_enabled = bool(openai_api_key)
            if self.semantic_enabled:
                logger.info("✅ Semantic search enabled with OpenAI embeddings API")
            else:
                logger.warning("⚠️ OpenAI API key not provided - using text search only")
        except Exception as e:
            logger.warning(f"❌ OpenAI client initialization failed: {e}")
            self.semantic_enabled = False
            
        # Thread pool for OpenAI API calls to avoid event loop conflicts
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="openai_embed")
    
    async def enable_semantic_search(self):
        """Enable semantic search capabilities."""
        if self.semantic_enabled:
            logger.info("✅ Semantic search already enabled with OpenAI")
        else:
            logger.warning("❌ Semantic search not available - OpenAI client not initialized")
        return self.semantic_enabled
        
    def init_memory_tables(self):
        """Initialize memory storage tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Memory entries table (updated for OpenAI embeddings)
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
                    openai_embedding BLOB,
                    importance INTEGER DEFAULT 5
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
    
    def _create_embedding_sync(self, text: str) -> bytes:
        """Create OpenAI embedding synchronously using thread pool."""
        if not self.semantic_enabled:
            return b''
        
        try:
            # Get embedding from OpenAI (runs in thread pool)
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",  # Cost-effective model
                input=text[:8000]  # Truncate to avoid token limits
            )
            embedding = response.data[0].embedding
            
            # Store as compact binary (much smaller than JSON)
            return struct.pack(f'{len(embedding)}f', *embedding)
            
        except Exception as e:
            logger.error(f"Error creating OpenAI embedding: {e}")
            return b''
    
    def _create_embedding(self, text: str) -> bytes:
        """Create embedding synchronously, handling event loop properly."""
        if not self.semantic_enabled:
            return b''
            
        try:
            # Use thread pool to avoid blocking event loop
            future = self._executor.submit(self._create_embedding_sync, text)
            return future.result(timeout=30)  # 30 second timeout
        except concurrent.futures.TimeoutError:
            logger.error("Embedding creation timed out")
            return b''
        except Exception as e:
            logger.error(f"Error in embedding creation: {e}")
            return b''
    
    async def _create_embedding_async(self, text: str) -> bytes:
        """Create embedding asynchronously for use in async contexts."""
        if not self.semantic_enabled:
            return b''
            
        try:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self._create_embedding_sync, text)
        except Exception as e:
            logger.error(f"Error in async embedding creation: {e}")
            return b''
    
    def _embedding_from_bytes(self, embedding_bytes: bytes) -> List[float]:
        """Convert binary embedding back to list."""
        if not self.semantic_enabled or not embedding_bytes:
            return None
        
        try:
            # Unpack binary data
            float_count = len(embedding_bytes) // 4
            return list(struct.unpack(f'{float_count}f', embedding_bytes))
        except Exception as e:
            logger.error(f"Error loading embedding: {e}")
            return None
    
    async def store_memory_async(self, content: str, user_id: int, channel_id: int, 
                                memory_type: str = 'interaction', tags: List[str] = None) -> str:
        """Store memory with OpenAI embedding (async version)."""
        try:
            memory_id = str(uuid.uuid4())
            
            # Generate embedding asynchronously
            embedding_bytes = await self._create_embedding_async(content) if self.semantic_enabled else b''
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO memories
                    (id, content, user_id, channel_id, timestamp, memory_type, tags, openai_embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory_id,
                    content,
                    user_id,
                    channel_id,
                    datetime.now().isoformat(),
                    memory_type,
                    json.dumps(tags or []),
                    embedding_bytes
                ))
            
            logger.info(f"🧠 Stored memory with OpenAI embedding: {content[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return ""

    def store_memory(self, content: str, user_id: int, channel_id: int, 
                    memory_type: str = 'interaction', tags: List[str] = None) -> str:
        """Store memory with OpenAI embedding (sync version with thread pool)."""
        try:
            memory_id = str(uuid.uuid4())
            
            # Generate embedding synchronously using thread pool
            embedding_bytes = self._create_embedding(content) if self.semantic_enabled else b''
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO memories
                    (id, content, user_id, channel_id, timestamp, memory_type, tags, openai_embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory_id,
                    content,
                    user_id,
                    channel_id,
                    datetime.now().isoformat(),
                    memory_type,
                    json.dumps(tags or []),
                    embedding_bytes
                ))
            
            if embedding_bytes:
                logger.info(f"🧠 Stored memory with OpenAI embedding: {content[:50]}...")
            else:
                logger.info(f"🧠 Stored memory (no embedding): {content[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return ""
    
    
    def query_memories(self, query: str, user_id: Optional[int] = None, 
                      memory_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Query memories using OpenAI embedding similarity."""
        if not self.semantic_enabled:
            # Fallback to text search
            return self._text_search(query, user_id, memory_type, limit)
        
        try:
            # Generate query embedding synchronously
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return self._text_search(query, user_id, memory_type, limit)
            
            query_vector = self._embedding_from_bytes(query_embedding)
            if not query_vector:
                return self._text_search(query, user_id, memory_type, limit)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get all memories with embeddings
                sql = '''
                    SELECT id, content, user_id, channel_id, timestamp, memory_type, tags, openai_embedding
                    FROM memories 
                    WHERE openai_embedding IS NOT NULL
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
                
                for row in cursor.fetchall():
                    try:
                        # Load memory embedding
                        memory_vector = self._embedding_from_bytes(row[7])
                        
                        if memory_vector:
                            # Calculate cosine similarity
                            similarity = self._cosine_similarity(query_vector, memory_vector)
                            
                            memories.append({
                                'id': row[0],
                                'content': row[1],
                                'user_id': row[2],
                                'channel_id': row[3],
                                'timestamp': row[4],
                                'memory_type': row[5],
                                'tags': json.loads(row[6]),
                                'similarity': similarity
                            })
                    except Exception as e:
                        logger.error(f"Error processing memory: {e}")
                        continue
                
                # Sort by similarity and return top results
                memories.sort(key=lambda x: x['similarity'], reverse=True)
                return memories[:limit]
                
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            # Fallback to text search
            return self._text_search(query, user_id, memory_type, limit)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity without numpy."""
        try:
            # Dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Magnitudes
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(a * a for a in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except:
            return 0.0
    
    def _text_search(self, query: str, user_id: Optional[int] = None, 
                    memory_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Fallback text-based search."""
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
                
                return [{
                    'id': row[0],
                    'content': row[1],
                    'user_id': row[2],
                    'channel_id': row[3],
                    'timestamp': row[4],
                    'memory_type': row[5],
                    'tags': json.loads(row[6]),
                    'similarity': 0.5  # Default similarity for text search
                } for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
    async def migrate_memories_to_openai(self):
        """Migrate existing memories to OpenAI embeddings."""
        if not self.semantic_enabled:
            logger.warning("OpenAI client not available for migration")
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get memories without OpenAI embeddings
                cursor = conn.execute('''
                    SELECT id, content FROM memories 
                    WHERE openai_embedding IS NULL 
                    AND content IS NOT NULL
                    AND content != ''
                ''')
                
                memories_to_update = cursor.fetchall()
                logger.info(f"🔄 Found {len(memories_to_update)} memories to migrate...")
                
                for i, (memory_id, content) in enumerate(memories_to_update):
                    try:
                        # Generate OpenAI embedding
                        embedding_bytes = await self._create_embedding(content)
                        
                        if embedding_bytes:
                            # Update memory with embedding
                            conn.execute('''
                                UPDATE memories 
                                SET openai_embedding = ?
                                WHERE id = ?
                            ''', (embedding_bytes, memory_id))
                            
                            if (i + 1) % 10 == 0:
                                logger.info(f"✅ Migrated {i + 1}/{len(memories_to_update)} memories")
                        
                        # Rate limit to avoid hitting OpenAI limits
                        await asyncio.sleep(0.1)  # 10 requests/second max
                        
                    except Exception as e:
                        logger.error(f"Error migrating memory {memory_id}: {e}")
                        continue
                
                logger.info("✅ Memory migration to OpenAI embeddings complete!")
                
        except Exception as e:
            logger.error(f"Error during migration: {e}")
    
    def get_memory_stats(self) -> Dict:
        """Get memory system statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count total memories
                cursor = conn.execute('SELECT COUNT(*) FROM memories')
                total_memories = cursor.fetchone()[0]
                
                # Count memories with OpenAI embeddings
                cursor = conn.execute('SELECT COUNT(*) FROM memories WHERE openai_embedding IS NOT NULL')
                memories_with_embeddings = cursor.fetchone()[0]
                
                # Count by type
                cursor = conn.execute('SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type')
                memory_types = dict(cursor.fetchall())
                
                return {
                    'total_memories': total_memories,
                    'memories_with_embeddings': memories_with_embeddings,
                    'embedding_coverage': f"{memories_with_embeddings}/{total_memories}",
                    'memory_types': memory_types,
                    'semantic_enabled': self.semantic_enabled
                }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def regenerate_embeddings_for_existing_memories(self, limit: int = 50) -> int:
        """Regenerate embeddings for existing memories that don't have them."""
        if not self.semantic_enabled:
            logger.warning("⚠️ Cannot regenerate embeddings - OpenAI API not configured")
            return 0
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Find memories without embeddings
                cursor = conn.execute('''
                    SELECT id, content FROM memories 
                    WHERE openai_embedding IS NULL OR openai_embedding = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (b'', limit))
                
                memories_to_update = cursor.fetchall()
                
            if not memories_to_update:
                logger.info("✅ All memories already have embeddings")
                return 0
                
            logger.info(f"🔄 Regenerating embeddings for {len(memories_to_update)} memories...")
            
            updated_count = 0
            for memory_id, content in memories_to_update:
                try:
                    # Generate new embedding synchronously
                    embedding_bytes = self._create_embedding(content)
                    
                    if embedding_bytes:
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute('''
                                UPDATE memories 
                                SET openai_embedding = ?
                                WHERE id = ?
                            ''', (embedding_bytes, memory_id))
                        
                        updated_count += 1
                        logger.info(f"🧠 Updated embedding for memory: {content[:50]}...")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to update embedding for memory {memory_id}: {e}")
                    continue
            
            logger.info(f"✅ Successfully regenerated {updated_count} embeddings")
            return updated_count
            
        except Exception as e:
            logger.error(f"❌ Error regenerating embeddings: {e}")
            return 0
    
    def cleanup(self):
        """Cleanup resources (thread pool)."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
            logger.info("🧹 Memory system thread pool cleaned up") 