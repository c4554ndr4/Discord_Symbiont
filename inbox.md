# Conversation History Persistence Plan for Dax

## Problem Analysis
Dax doesn't maintain conversation history between interactions because:

1. **In-Memory Storage Only**: `ConversationManager.recent_contexts` is a Python dict that gets cleared on bot restart
2. **No Persistence Layer**: Conversation context isn't saved to disk/database
3. **Bot Restart Impact**: Each restart wipes all conversation history 
4. **Limited Context Window**: Only last 6 messages kept (though this is reasonable)

## Current System Overview
- ✅ **Context Building**: Properly builds conversation messages with recent history
- ✅ **Message Addition**: Correctly adds user/assistant messages to context
- ✅ **Memory Integration**: Long-term memory system works via embeddings
- ❌ **Persistence**: No conversation state persistence between sessions

## Solution Plan: Persistent Conversation Context

### Phase 1: Add Conversation Persistence (Priority: HIGH)

**1.1 Database Schema Addition**
```sql
-- Add to existing memory database
CREATE TABLE conversation_contexts (
    context_key TEXT PRIMARY KEY,  -- "dm_{user_id}" or "channel_{channel_id}"
    messages TEXT NOT NULL,        -- JSON array of recent messages
    last_updated TEXT NOT NULL,    -- ISO timestamp
    message_count INTEGER DEFAULT 0
);
```

**1.2 Enhanced ConversationManager**
```python
class ConversationManager:
    def __init__(self, db_path: str = "bot_memory.db"):
        self.db_path = db_path
        self.recent_contexts = {}  # In-memory cache
        self.recent_message_limit = 6
        self._load_contexts_from_db()
    
    def _load_contexts_from_db(self):
        """Load conversation contexts from database on startup"""
        
    def _save_context_to_db(self, context_key: str):
        """Save conversation context to database"""
        
    def add_message_to_context(self, message, role: str, content: str):
        """Enhanced to persist to database"""
        # Current logic + database save
```

**1.3 Context Cleanup Strategy**
- Conversations older than 7 days auto-deleted
- Implement LRU cache for in-memory contexts
- Periodic cleanup task to prevent database bloat

### Phase 2: Improved Context Management (Priority: MEDIUM)

**2.1 Smarter Context Limits**
- Increase recent context to 10-12 messages for DMs 
- Keep 6 messages for busy channels
- Token-aware context trimming (keep more short messages)

**2.2 Context Quality Enhancements**
```python
def get_conversation_summary(self, context_key: str) -> str:
    """Generate conversation summary for long sessions"""
    # Use cheap model to summarize older context
    # Include summary in system message for continuity
```

**2.3 Cross-Channel Context Awareness**
- Track if user recently talked in other channels
- Include brief cross-channel context in system message
- "You recently discussed X with this user in #other-channel"

### Phase 3: OpenRouter-Specific Optimizations (Priority: LOW)

**3.1 Conservative Token Management**
- Implement token counting for conversation context
- Prioritize recent messages when context exceeds limits
- Smart truncation at natural conversation boundaries

**3.2 Stop Sequence Integration**
```python
# In conversation context building
def build_conversation_messages(self, message, content: str):
    messages = self._get_recent_context_with_summaries()
    # Include conversation-aware stop sequences
    return messages, ["---", "## New Topic", "\n\n---"]
```

### Phase 4: Advanced Features (Priority: FUTURE)

**4.1 Conversation Threading**
- Detect topic changes in conversations
- Maintain separate context threads
- Allow "going back" to previous topics

**4.2 User Preference Integration** 
```yaml
# In config.yaml
conversation:
  per_user_context_limits:
    default: 6
    power_users: 12
  context_persistence_days: 7
  enable_conversation_summaries: true
```

## Implementation Priority

### Immediate (This Week)
1. **Add conversation_contexts table** to memory database
2. **Enhance ConversationManager** with persistence
3. **Test with Cassandra's DMs** to verify continuity

### Short Term (Next Week)  
1. Increase context limits for DMs
2. Add conversation cleanup task
3. Optimize for OpenRouter token limits

### Long Term (Future)
1. Conversation summaries for long sessions
2. Cross-channel context awareness  
3. Advanced threading and topic detection

## Expected Impact
- ✅ **Immediate Memory**: Dax remembers previous messages in conversation
- ✅ **Restart Resilience**: Context persists through bot restarts
- ✅ **Natural Flow**: Conversations feel continuous and natural
- ✅ **Efficient**: Minimal performance impact with smart caching

## Testing Strategy
1. **DM Test**: Multi-session conversation with Cassandra
2. **Channel Test**: Busy channel conversation continuity  
3. **Restart Test**: Verify persistence through bot restart
4. **Load Test**: Performance with many active conversations

This plan will solve the "fresh start" problem and make Dax feel like it truly remembers conversations! 🧠✨
