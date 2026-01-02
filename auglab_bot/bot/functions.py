"""
Function Implementations for Augmentation Lab Bot
Contains all function implementations that can be called by the AI
"""

import os
import json
import sqlite3
import asyncio
import logging
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from ..storage.cost_tracker import CostEntry

logger = logging.getLogger(__name__)

class FunctionImplementations:
    """Contains all function implementations that can be called by the AI."""
    
    def __init__(self, cost_tracker, memory_system, working_memory_manager, 
                 model_manager, config_manager, data_store, db_path):
        self.cost_tracker = cost_tracker
        self.memory_system = memory_system
        self.working_memory_manager = working_memory_manager
        self.model_manager = model_manager
        self.config = config_manager
        self.data_store = data_store
        self.db_path = db_path
        
        # Autonomous mode state
        self.autonomous_mode_active = False
        self.autonomous_tool_calls_used = 0
        self.autonomous_context_history = []
        
        # Pending messages for autonomous mode
        self._pending_autonomous_message = None
        
        # Bot reference (set after initialization)
        self.bot = None
        
        # Importance criteria
        self.importance_criteria = self._get_default_importance_criteria()
    
    def _get_default_importance_criteria(self) -> str:
        """Get default importance criteria for message assessment."""
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
    
    def update_constitution(self, new_constitution: str, reason: str = "") -> Dict[str, Any]:
        """Update the bot's constitution."""
        success = self.working_memory_manager.update_working_memory(new_constitution, reason)
        return {
            "success": success,
            "message": "Constitution updated successfully" if success else "Constitution update failed or no change needed"
        }
    
    def query_memory(self, query: str, user_id: int = None, limit: int = 10) -> Dict[str, Any]:
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
    
    def get_user_history(self, user_id: int, limit: int = 20) -> Dict[str, Any]:
        """Get interaction history for a specific user."""
        history = self.memory_system.get_user_interaction_history(user_id, limit)
        return {
            "history": history,
            "count": len(history),
            "user_id": user_id
        }
    
    def edit_memory(self, memory_id: str, new_content: str, reason: str = "") -> Dict[str, Any]:
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
    
    def delete_memory(self, memory_id: str, reason: str = "") -> Dict[str, Any]:
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
    
    def store_observation(self, content: str, user_id: int = 0, channel_id: int = 0, tags: str = "") -> Dict[str, Any]:
        """Store an observation or reflection in memory."""
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
        memory_id = self.memory_system.store_memory(
            content, user_id, channel_id, 
            memory_type='observation', tags=tag_list
        )
        return {
            "success": bool(memory_id),
            "memory_id": memory_id,
            "message": "Observation stored successfully" if memory_id else "Failed to store observation"
        }
    
    def get_working_memory(self) -> Dict[str, Any]:
        """Get the current working memory."""
        working_memory = self.working_memory_manager.get_current_working_memory()
        return {
            "working_memory": working_memory,
            "length": len(working_memory)
        }
    
    def debug_system_status(self) -> Dict[str, Any]:
        """Comprehensive system debug information."""
        import os
        from datetime import datetime
        
        try:
            # Get memory system stats
            memory_stats = self.memory_system.get_memory_stats()
            
            # Get API key status (masked)
            api_keys = {}
            for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'OPEN_ROUTER_KEY', 'GROQ_API_KEY']:
                value = os.getenv(key)
                if value:
                    api_keys[key] = f"{value[:8]}...{value[-8:]}" if len(value) > 16 else "***SET***"
                else:
                    api_keys[key] = "NOT_SET"
            
            # Get recent logs if available
            recent_logs = []
            error_logs = []
            if hasattr(self, 'bot') and hasattr(self.bot, 'log_capture'):
                recent_logs = self.bot.log_capture.get_recent_logs(10)
                error_logs = self.bot.log_capture.get_error_logs(5)
            
            # Test embedding generation
            embedding_test = "✅ Working"
            try:
                if self.memory_system.semantic_enabled:
                    test_embedding = self.memory_system._create_embedding("test")
                    if not test_embedding:
                        embedding_test = "❌ Failed to generate"
                else:
                    embedding_test = "⚠️ Disabled (no API key)"
            except Exception as e:
                embedding_test = f"❌ Error: {str(e)[:50]}"
            
            # Bot configuration
            bot_config = {
                "response_mode": self.config.get('bot.response_mode', 'mentions_only'),
                "residency_role": self.config.get('bot.residency_role_name', "Resident '25"),
                "semantic_enabled": self.memory_system.semantic_enabled,
                "conversation_memory": self.config.get('features.conversation_memory', True),
                "debug": self.config.get('debug', False)
            }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_status": "operational",
                "memory_stats": memory_stats,
                "embedding_test": embedding_test,
                "api_keys": api_keys,
                "bot_config": bot_config,
                "recent_logs": [f"[{log['level']}] {log['message'][:100]}" for log in recent_logs[-5:]],
                "error_logs": [f"[{log['level']}] {log['message'][:100]}" for log in error_logs[-3:]],
                "uptime": "Available via bot.uptime if tracked"
            }
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "system_status": f"error: {str(e)}",
                "error": str(e)
            }
    
    def think(self, content: str) -> Dict[str, Any]:
        """Internal thinking/reflection function - not shown to users."""
        logger.info(f"🤔 Bot thinking: {content[:100]}...")
        return {"thought_processed": True, "length": len(content)}
    
    def message_user(self, message: str, users: List[str] = None, name: str = "", user_id: int = 0, destination: str = "current", context: str = "") -> Dict[str, Any]:
        """
        Unified messaging function - handles all user messaging scenarios.
        
        Args:
            message: The message content to send
            users: List of user names or IDs to tag (can be empty for no tags)
            name: Find user by this name (for backwards compatibility)
            user_id: Use this user ID directly (for backwards compatibility)
            destination: Where to send the message:
                        - "dm" = send direct message (only works with single user)
                        - "current" = tag user(s) in current channel (default)
                        - channel name like "chat-with-dax" or "🌎-res-general" = send to that channel
            context: Additional context for the message
            
        Returns:
            Dict with success status and details
        """
        if not self.bot:
            return {
                "success": False,
                "error": "Bot reference not available",
                "message": "Cannot send message - bot not initialized"
            }
            
        try:
            # Handle backwards compatibility: convert single name/user_id to users list
            if users is None:
                users = []
                if name:
                    users.append(name)
                elif user_id:
                    users.append(str(user_id))
            
            # Step 1: Find all target users
            target_users = []
            if users:
                for user_identifier in users:
                    target_user = None
                    
                    # Try as user ID first (if it's numeric)
                    if str(user_identifier).isdigit():
                        user_id_int = int(user_identifier)
                        target_user = self.bot.get_user(user_id_int)
                        if not target_user:
                            # Try to fetch from API
                            import asyncio
                            loop = asyncio.get_event_loop()
                            try:
                                target_user = loop.run_until_complete(self.bot.fetch_user(user_id_int))
                            except:
                                continue  # Skip this user if not found
                    else:
                        # Try as name
                        find_result = self.find_member_by_name(str(user_identifier))
                        if find_result.get("success", False):
                            members = find_result.get("members", [])
                            if members:
                                member = members[0]
                                user_id_int = member.get("id")
                                target_user = self.bot.get_user(user_id_int)
                                if not target_user:
                                    # Try to fetch from API
                                    import asyncio
                                    loop = asyncio.get_event_loop()
                                    try:
                                        target_user = loop.run_until_complete(self.bot.fetch_user(user_id_int))
                                    except:
                                        continue  # Skip this user if not found
                    
                    if target_user:
                        target_users.append(target_user)
            
            # Step 2: Handle different destinations
            if destination == "dm":
                # DMs only work with exactly one user
                if len(target_users) != 1:
                    return {
                        "success": False,
                        "error": "DMs require exactly one user",
                        "message": f"Cannot send DM to {len(target_users)} users. DMs work with exactly one user."
                    }
                
                target_user = target_users[0]
                self._pending_dm = {
                    "user": target_user,
                    "message": message[:2000],  # Discord limit
                    "context": context,
                    "timestamp": datetime.now()
                }
                
                return {
                    "success": True,
                    "user_count": 1,
                    "users": [{"user_id": target_user.id, "username": target_user.name, "display_name": getattr(target_user, 'display_name', target_user.name)}],
                    "destination": "DM",
                    "message_length": len(message),
                    "message": f"DM queued for {target_user.name}: {message[:100]}..."
                }
                
            elif destination == "current":
                # Tag users in current channel (return tags for inclusion in response)
                user_tags = [f"<@{user.id}>" for user in target_users]
                
                return {
                    "success": True,
                    "user_count": len(target_users),
                    "user_tags": user_tags,
                    "users": [{"user_id": user.id, "username": user.name, "display_name": getattr(user, 'display_name', user.name)} for user in target_users],
                    "destination": "current channel",
                    "context": message,
                    "message": f"Tagged {len(target_users)} user(s): {', '.join([getattr(user, 'display_name', user.name) for user in target_users])}" if target_users else "No users tagged - sending general message"
                }
                
            else:
                # Send to specific channel
                target_guild_id = self.config.get('bot.target_guild_id')
                guild = self.bot.get_guild(target_guild_id) if target_guild_id else None
                
                if not guild:
                    return {
                        "success": False,
                        "error": "Guild not found",
                        "message": "Cannot access the target server"
                    }
                
                # Look for channel by name (handle emoji prefixes)
                channel = None
                for c in guild.channels:
                    if hasattr(c, 'name'):
                        # Try exact match first
                        if c.name == destination:
                            channel = c
                            break
                        # Try matching without emoji prefix (e.g., "🌎-res-general" -> "res-general")
                        if destination.startswith('🌎-') and c.name == destination[2:]:
                            channel = c
                            break
                        # Try other common emoji prefixes
                        for prefix in ['🌎-', '💬-', '📢-', '🔧-', '🎯-']:
                            if destination.startswith(prefix) and c.name == destination[len(prefix):]:
                                channel = c
                                break
                
                if not channel:
                    return {
                        "success": False,
                        "error": f"Channel '#{destination}' not found",
                        "message": f"Could not find channel #{destination} in the server"
                    }
                
                # Check channel access using the new system
                access_check = self.check_channel_access(destination, "user_messaging")
                if not access_check["success"]:
                    allowed_channels = self.config.get('channels.user_messaging_allowed', ["chat-with-dax", "🌎-res-general"])
                    return {
                        "success": False,
                        "error": f"Channel not allowed for user requests",
                        "message": f"{access_check['message']}. Available channels: {', '.join(allowed_channels)}"
                    }
                
                # Include user tags in the message for channel sends (if any users specified)
                if target_users:
                    user_tags = [f"<@{user.id}>" for user in target_users]
                    full_message = f"{' '.join(user_tags)} {message}"
                else:
                    full_message = message
                
                # Store the message for sending
                self._pending_user_message = {
                    "channel": channel,
                    "message": full_message[:2000],  # Discord limit
                    "requester": context,
                    "timestamp": datetime.now()
                }
                
                return {
                    "success": True,
                    "user_count": len(target_users),
                    "users": [{"user_id": user.id, "username": user.name, "display_name": getattr(user, 'display_name', user.name)} for user in target_users],
                    "channel_id": channel.id,
                    "channel_name": channel.name,
                    "destination": f"#{destination}",
                    "message_length": len(full_message),
                    "message": f"Message queued for #{channel.name} with {len(target_users)} user tag(s): {full_message[:100]}..." if target_users else f"General message queued for #{channel.name}: {full_message[:100]}..."
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error sending message: {str(e)}"
            }

    # DEPRECATED FUNCTIONS - keeping for backwards compatibility but marking as deprecated
    def find_and_tag_user(self, name: str, context: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use message_user with destination='current' instead."""
        return self.message_user(message=context, name=name, destination="current")
    
    def send_message_to_channel(self, channel_name: str, message: str, requester_name: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use message_user with destination=channel_name instead."""
        # For channel-only messages without user tagging, we'll keep this simpler version
        if not self.bot:
            return {
                "success": False,
                "error": "Bot reference not available",
                "message": "Cannot send message - bot not initialized"
            }
        
        try:
            target_guild_id = self.config.get('bot.target_guild_id')
            guild = self.bot.get_guild(target_guild_id) if target_guild_id else None
            
            if not guild:
                return {
                    "success": False,
                    "error": "Guild not found",
                    "message": "Cannot access the target server"
                }
            
            # Look for channel by name
            channel = None
            for c in guild.channels:
                if hasattr(c, 'name') and c.name == channel_name:
                    channel = c
                    break
            
            if not channel:
                return {
                    "success": False,
                    "error": f"Channel '#{channel_name}' not found",
                    "message": f"Could not find channel #{channel_name} in the server"
                }
            
            # Check if this is an allowed channel for messaging
            allowed_channels = self.config.get('channels.user_messaging_allowed', ["chat-with-dax", "🌎-res-general"])
            if channel_name not in allowed_channels:
                return {
                    "success": False,
                    "error": f"Channel not allowed for user requests",
                    "message": f"I can only send messages to: {', '.join(allowed_channels)}"
                }
            
            # Store the message for sending
            self._pending_user_message = {
                "channel": channel,
                "message": message[:2000],  # Discord limit
                "requester": requester_name,
                "timestamp": datetime.now()
            }
            
            return {
                "success": True,
                "channel_id": channel.id,
                "channel_name": channel.name,
                "message_length": len(message),
                "message": f"Message queued for #{channel.name}: {message[:100]}..."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error queueing message: {str(e)}"
            }
    
    def tag_user(self, user_id: int, context: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use message_user with user_id and destination='current' instead."""
        return self.message_user(message=context, user_id=user_id, destination="current")
    
    def send_direct_message(self, user_id: int, message: str, context: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use message_user with user_id and destination='dm' instead."""
        return self.message_user(message=message, user_id=user_id, destination="dm", context=context)
    
    def find_and_dm_user(self, name: str, message: str, context: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use message_user with name and destination='dm' instead."""
        return self.message_user(message=message, name=name, destination="dm", context=context)
    
    def react_to_message(self, emoji: str, context: str = "") -> Dict[str, Any]:
        """React to the current message with an emoji."""
        if not self.bot or not hasattr(self, '_current_message') or not self._current_message:
            return {
                "success": False,
                "error": "No message to react to",
                "message": "Cannot react - no current message available"
            }
        
        try:
            # Store the reaction for processing
            self._pending_reaction = {
                "message": self._current_message,
                "emoji": emoji,
                "context": context,
                "timestamp": datetime.now()
            }
            
            return {
                "success": True,
                "emoji": emoji,
                "context": context,
                "pending": True,
                "message": f"Will react with {emoji}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to prepare reaction with {emoji}"
            }
    
    def google_search(self, query: str) -> Dict[str, Any]:
        """Search web for strategic information (using Perplexity API)."""
        try:
            import requests
            
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
    
    def update_importance_criteria(self, criteria: str, reason: str = "") -> Dict[str, Any]:
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
            """, (criteria, reason))
            
            conn.commit()
            conn.close()
            
            self.importance_criteria = criteria
            logger.info(f"Updated importance criteria: {reason}")
            
            return {
                "success": True,
                "new_criteria": criteria,
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
                return self.importance_criteria
                
        except Exception as e:
            logger.error(f"Error getting importance criteria: {e}")
            return self.importance_criteria
    
    def generate_image(self, prompt: str, quality: str = "medium", size: str = "1024x1024") -> Dict[str, Any]:
        """Generate an image using OpenAI's DALL-E. Cost varies by quality."""
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
            response = self.model_manager.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            # Calculate cost
            if quality == "low":
                actual_cost = 0.040  # Standard pricing
            elif quality == "high":
                actual_cost = 0.080  # HD pricing
            else:  # medium
                actual_cost = 0.040  # Standard pricing
            
            # Record the usage
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model='dall-e-3',
                input_tokens=len(prompt.split()) * 1,
                output_tokens=1,
                cost=actual_cost,
                action='image_generation_function',
                user_id=0,
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
    
    def run_command(self, command: str, timeout: int = 30, language: str = "bash") -> Dict[str, Any]:
        """Execute command line tools safely in Docker sandbox."""
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
    
    def create_script(self, content: str, filename: str = "script.py", language: str = "python") -> Dict[str, Any]:
        """Create and execute scripts for complex tasks."""
        try:
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
    
    def get_budget(self) -> Dict[str, Any]:
        """Get current budget status and spending breakdown."""
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
                    "warning": "Low budget!" if report['remaining'] < 1.0 else None
                },
                "message": f"Budget: ${report['remaining']:.2f} remaining of ${report['budget']:.2f} ({percentage_used:.1f}% used)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting budget info: {str(e)}"
            }
    
    def get_current_time(self) -> Dict[str, Any]:
        """Get current time and calculate days until MIT Media Lab presentation."""
        try:
            now = datetime.now(timezone.utc)
            
            # MIT Media Lab presentation deadline
            presentation_date = datetime(2025, 8, 23, 12, 0, 0, tzinfo=timezone.utc)
            
            days_until_presentation = (presentation_date - now).days
            
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
                    "urgency_level": urgency,
                    "time_status": time_status,
                    "is_weekend": now.weekday() >= 5
                },
                "message": f"Current time: {now.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} | {time_status}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting time info: {str(e)}"
            }
    
    def send_message(self, channel_id=None, message: str = "", channel_name: str = "") -> Dict[str, Any]:
        """Send a message to a specific channel (autonomous mode only)."""
        if not self.autonomous_mode_active:
            return {
                "success": False,
                "error": "send_message can only be used in autonomous mode",
                "message": "This function is restricted to autonomous operations"
            }
        
        if not self.bot:
            return {
                "success": False,
                "error": "Bot reference not available",
                "message": "Cannot send message - bot not initialized"
            }
            
        try:
            channel = None
            
            # Try to find channel by ID first, then by name
            if channel_id and str(channel_id).isdigit():
                channel = self.bot.get_channel(int(channel_id))
            
            # If no channel found by ID, try to find by name
            if not channel and channel_name:
                target_guild_id = self.config.get('bot.target_guild_id')
                guild = self.bot.get_guild(target_guild_id) if target_guild_id else None
                
                if guild:
                    for c in guild.channels:
                        if hasattr(c, 'name') and c.name == channel_name:
                            channel = c
                            break
            
            # If still no channel and channel_id looks like a name, try that
            if not channel and channel_id and not str(channel_id).isdigit():
                target_guild_id = self.config.get('bot.target_guild_id')
                guild = self.bot.get_guild(target_guild_id) if target_guild_id else None
                
                if guild:
                    for c in guild.channels:
                        if hasattr(c, 'name') and c.name == str(channel_id):
                            channel = c
                            break
            
            if not channel:
                return {
                    "success": False,
                    "error": f"Channel not found",
                    "message": f"Could not find channel {channel_name or channel_id}. Use get_available_channels to see options."
                }
            
            # Check if channel restrictions are enabled and validate channel
            if self.config.get('autonomous.channel_restrictions', True):
                allowed_channels = self.config.get('channels.autonomous_allowed', 
                                                   self.config.get('autonomous.allowed_channels', []))
                if channel.name not in allowed_channels:
                    available_channels = ", ".join(allowed_channels)
                    return {
                        "success": False,
                        "error": "Channel not allowed",
                        "message": f"Channel #{channel.name} not available. Use one of these channels: {available_channels}"
                    }
            
            # Store the message task for execution
            self._pending_autonomous_message = {
                "channel": channel,
                "message": message[:2000],  # Discord limit
                "timestamp": datetime.now()
            }
            
            return {
                "success": True,
                "channel_id": channel.id,
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
    
    def get_available_channels(self) -> Dict[str, Any]:
        """Get list of available channels for autonomous mode."""
        try:
            if not self.bot:
                return {
                    "success": False,
                    "error": "Bot reference not available"
                }
            
            allowed_channels = self.config.get('channels.autonomous_allowed', 
                                               self.config.get('autonomous.allowed_channels', []))
            channel_info = []
            
            # Get target guild
            target_guild_id = self.config.get('bot.target_guild_id')
            guild = self.bot.get_guild(target_guild_id) if target_guild_id else None
            
            if guild:
                for channel_name in allowed_channels:
                    # Find channel by name
                    channel = None
                    for c in guild.channels:
                        if hasattr(c, 'name') and c.name == channel_name:
                            channel = c
                            break
                    
                    if channel:
                        channel_info.append({
                            "name": channel.name,
                            "id": channel.id,
                            "type": str(channel.type)
                        })
                    else:
                        channel_info.append({
                            "name": channel_name,
                            "id": "not_found",
                            "type": "unknown"
                        })
            
            return {
                "success": True,
                "allowed_channels": allowed_channels,
                "channel_details": channel_info,
                "message": f"Available channels: {', '.join(allowed_channels)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting available channels: {str(e)}"
            }
    
    def get_residency_members(self) -> Dict[str, Any]:
        """Get list of Residency '25 members with their handles."""
        try:
            if not self.bot:
                return {
                    "success": False,
                    "error": "Bot reference not available"
                }
            
            # Get target guild
            target_guild_id = self.config.get('bot.target_guild_id')
            guild = self.bot.get_guild(target_guild_id) if target_guild_id else None
            
            if not guild:
                return {
                    "success": False,
                    "error": "Target guild not found"
                }
            
            # Get residency role name
            role_name = self.config.get('bot.residency_role_name', "Resident '25")
            residency_role = None
            for role in guild.roles:
                if role.name == role_name:
                    residency_role = role
                    break
            
            if not residency_role:
                return {
                    "success": False,
                    "error": f"Role '{role_name}' not found in guild"
                }
            
            # Get members with residency role
            members = []
            for member in guild.members:
                if residency_role in member.roles:
                    members.append({
                        "display_name": member.display_name,
                        "username": member.name,
                        "id": member.id,
                        "mention": f"<@{member.id}>",
                        "status": str(member.status) if hasattr(member, 'status') else 'unknown'
                    })
            
            return {
                "success": True,
                "role_name": role_name,
                "member_count": len(members),
                "members": members,
                "message": f"Found {len(members)} Residency '25 members"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting residency members: {str(e)}"
            }
    
    def find_member_by_name(self, name: str) -> Dict[str, Any]:
        """Find a member by name (display name or username)."""
        try:
            if not self.bot:
                return {
                    "success": False,
                    "error": "Bot reference not available"
                }
            
            # Get target guild
            target_guild_id = self.config.get('bot.target_guild_id')
            guild = self.bot.get_guild(target_guild_id) if target_guild_id else None
            
            if not guild:
                return {
                    "success": False,
                    "error": "Target guild not found"
                }
            
            # Search for member by name (case insensitive)
            name_lower = name.lower()
            found_members = []
            
            for member in guild.members:
                if (name_lower in member.display_name.lower() or 
                    name_lower in member.name.lower() or
                    name_lower == member.display_name.lower() or
                    name_lower == member.name.lower()):
                    found_members.append({
                        "display_name": member.display_name,
                        "username": member.name,
                        "id": member.id,
                        "mention": f"<@{member.id}>",
                        "status": str(member.status) if hasattr(member, 'status') else 'unknown'
                    })
            
            if not found_members:
                return {
                    "success": False,
                    "error": f"No member found with name '{name}'"
                }
            
            return {
                "success": True,
                "search_name": name,
                "found_count": len(found_members),
                "members": found_members,
                "message": f"Found {len(found_members)} member(s) matching '{name}'"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error finding member: {str(e)}"
            }

    def stop_autonomous_mode(self, reason: str = "Task completed") -> Dict[str, Any]:
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
    
    def trigger_autonomous_mode(self, reason: str = "AI-triggered strategic analysis") -> Dict[str, Any]:
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
        
        # Set autonomous mode active
        self.autonomous_mode_active = True
        
        return {
            "success": True,
            "reason": reason,
            "message": f"Autonomous mode triggered: {reason}. Strategic operations will begin shortly."
        }
    
    def summarize_context(self, context_type: str = "all") -> Dict[str, Any]:
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
            if self.bot and hasattr(self.bot, 'target_guild_id') and self.bot.target_guild_id:
                guild = self.bot.get_guild(self.bot.target_guild_id)
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
    
    def stop(self, reason: str = "") -> str:
        """Stop function processing and return final response.
        
        Args:
            reason: Optional reason for stopping
            
        Returns:
            Confirmation message
        """
        try:
            logger.info(f"🛑 Stop function called: {reason}")
            return f"Function processing completed. {reason}".strip()
            
        except Exception as e:
            return f"Error in stop function: {str(e)}"
    
    def get_all_functions(self) -> Dict[str, Any]:
        """Get all available functions for registration."""
        return {
            'update_constitution': self.update_constitution,
            'query_memory': self.query_memory,
            'get_user_history': self.get_user_history,
            'store_observation': self.store_observation,
            'get_working_memory': self.get_working_memory,
            'debug_system_status': self.debug_system_status,
            'think': self.think,
            'message_user': self.message_user,  # UNIFIED MESSAGING FUNCTION
            'check_channel_access': self.check_channel_access,
            'list_channel_access': self.list_channel_access,
            'google_search': self.google_search,
            'update_importance_criteria': self.update_importance_criteria,
            'run_command': self.run_command,
            'create_script': self.create_script,
            'generate_image': self.generate_image,
            'get_budget': self.get_budget,
            'get_current_time': self.get_current_time,
            'send_message': self.send_message,
            'get_available_channels': self.get_available_channels,
            'get_residency_members': self.get_residency_members,
            'find_member_by_name': self.find_member_by_name,
            'react_to_message': self.react_to_message,
            'stop_autonomous_mode': self.stop_autonomous_mode,
            'summarize_context': self.summarize_context,
            'trigger_autonomous_mode': self.trigger_autonomous_mode,
            'edit_memory': self.edit_memory,
            'delete_memory': self.delete_memory,
            'stop': self.stop
            # REMOVED DEPRECATED MESSAGING FUNCTIONS:
            # - find_and_tag_user (use message_user with destination="current")
            # - send_message_to_channel (use message_user with destination=channel_name)
            # - tag_user (use message_user with user_id and destination="current")  
            # - send_direct_message (use message_user with user_id and destination="dm")
            # - find_and_dm_user (use message_user with name and destination="dm")
        } 

    def check_channel_access(self, channel_name: str, access_type: str = "user_messaging", user_roles: List[str] = None) -> Dict[str, Any]:
        """
        Check if the bot has access to a specific channel for a given purpose.
        
        Args:
            channel_name: Name of the channel to check
            access_type: Type of access needed ('user_messaging', 'autonomous', 'read_only')
            user_roles: List of user role names (for residents-only enforcement)
        
        Returns:
            Dict with success status and details
        """
        try:
            user_roles = user_roles or []
            
            # Check restricted channels first
            restricted_channels = self.config.get('channels.restricted', [])
            if channel_name in restricted_channels:
                return {
                    "success": False,
                    "reason": "restricted",
                    "message": f"Channel #{channel_name} is restricted"
                }
            
            # Check residents-only channels
            residents_only = self.config.get('channels.access_rules.residents_only_channels', [])
            residency_role = self.config.get('bot.residency_role_name', "Resident '25")
            if channel_name in residents_only and residency_role not in user_roles:
                return {
                    "success": False,
                    "reason": "residents_only",
                    "message": f"Channel #{channel_name} requires {residency_role} role"
                }
            
            # Check specific access type
            config_key = f'channels.{access_type}_allowed'
            allowed_channels = self.config.get(config_key, [])
            
            if channel_name in allowed_channels:
                return {
                    "success": True,
                    "reason": "explicitly_allowed",
                    "message": f"Access granted to #{channel_name}"
                }
            
            # Check if it's a project channel and project help is enabled
            if access_type == "user_messaging":
                project_channels = self.config.get('channels.project_channels', [])
                project_help_auto = self.config.get('channels.access_rules.project_help_auto_access', False)
                if channel_name in project_channels and project_help_auto:
                    return {
                        "success": True,
                        "reason": "project_help",
                        "message": f"Project help access granted to #{channel_name}"
                    }
            
            # Check personal channels
            personal_channels = self.config.get('channels.personal_channels', [])
            if channel_name in personal_channels:
                requires_permission = self.config.get('channels.access_rules.personal_requires_permission', True)
                if requires_permission:
                    return {
                        "success": False,
                        "reason": "personal_permission_required",
                        "message": f"Channel #{channel_name} requires explicit permission"
                    }
                else:
                    return {
                        "success": True,
                        "reason": "personal_allowed",
                        "message": f"Personal channel access granted to #{channel_name}"
                    }
            
            # Apply default action
            default_action = self.config.get('channels.access_rules.default_action', 'deny')
            if default_action == "allow":
                return {
                    "success": True,
                    "reason": "default_allow",
                    "message": f"Default allow policy grants access to #{channel_name}"
                }
            else:
                return {
                    "success": False,
                    "reason": "default_deny",
                    "message": f"Channel #{channel_name} not in allowed list"
                }
                
        except Exception as e:
            return {
                "success": False,
                "reason": "error",
                "message": f"Error checking channel access: {str(e)}"
            } 

    def list_channel_access(self) -> Dict[str, Any]:
        """
        List all channel categories and their access rules.
        Useful for users to understand where Dax can operate.
        """
        try:
            channel_info = {
                "user_messaging_allowed": {
                    "description": "Channels where users can ask Dax to send messages",
                    "channels": self.config.get('channels.user_messaging_allowed', [])
                },
                "autonomous_allowed": {
                    "description": "Channels where Dax can operate autonomously", 
                    "channels": self.config.get('channels.autonomous_allowed', [])
                },
                "read_only": {
                    "description": "Channels Dax can monitor but not send messages to",
                    "channels": self.config.get('channels.read_only', [])
                },
                "project_channels": {
                    "description": "Project-specific channels (may allow help requests)",
                    "channels": self.config.get('channels.project_channels', [])
                },
                "social_channels": {
                    "description": "Social/community channels", 
                    "channels": self.config.get('channels.social_channels', [])
                },
                "personal_channels": {
                    "description": "Personal/individual channels (permission required)",
                    "channels": self.config.get('channels.personal_channels', [])
                },
                "restricted": {
                    "description": "Restricted channels (no access)",
                    "channels": self.config.get('channels.restricted', [])
                },
                "access_rules": {
                    "default_action": self.config.get('channels.access_rules.default_action', 'deny'),
                    "project_help_auto_access": self.config.get('channels.access_rules.project_help_auto_access', False),
                    "personal_requires_permission": self.config.get('channels.access_rules.personal_requires_permission', True),
                    "residents_only_channels": self.config.get('channels.access_rules.residents_only_channels', [])
                }
            }
            
            return {
                "success": True,
                "channel_info": channel_info,
                "message": "Channel access information retrieved successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error retrieving channel access info: {str(e)}"
            } 