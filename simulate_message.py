#!/usr/bin/env python3
"""
Simple message simulator for testing dax Discord bot functionality.
Sends mock DM messages to dax as if they're coming from a specific user.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auglab_bot.ai.response_generator import ResponseGenerator
from auglab_bot.config.manager import ConfigManager
from auglab_bot.storage.cost_tracker import CostTracker  
from auglab_bot.storage.memory_system import MemorySystem
from auglab_bot.storage.data_store import ChannelDataStore
from auglab_bot.management.constitution import ConstitutionManager
from auglab_bot.ai.model_manager import ModelManager
from auglab_bot.ai.function_calling import FunctionCallManager
from auglab_bot.ai.conversation_manager import ConversationManager
from auglab_bot.bot.functions import FunctionImplementations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockUser:
    """Mock Discord user for testing."""
    def __init__(self, user_id: int, username: str, display_name: str):
        self.id = user_id
        self.name = username
        self.display_name = display_name
        
class MockChannel:
    """Mock Discord DM channel for testing.""" 
    def __init__(self, channel_id: int = 123456789):
        self.id = channel_id
        
class MockMessage:
    """Mock Discord message for testing."""
    def __init__(self, content: str, user_id: int = 314159, username: str = "m314x", display_name: str = "Test User"):
        self.content = content
        self.author = MockUser(user_id, username, display_name)
        self.channel = MockChannel()
        self.created_at = datetime.now()

async def simulate_message(message_content: str, user_id: int = 314159, username: str = "m314x", display_name: str = "Test User"):
    """Simulate sending a message to dax and get response."""
    try:
        # Initialize components
        config = ConfigManager()
        cost_tracker = CostTracker(
            budget=5.0,  # Default budget
            db_path="./auglab_bot.db"  # Default database path
        )
        memory_system = MemorySystem("./auglab_bot.db")
        data_store = ChannelDataStore("./auglab_bot.db")
        constitution_manager = ConstitutionManager("./auglab_bot.db")
        model_manager = ModelManager(config)
        function_manager = FunctionCallManager()
        conversation_manager = ConversationManager()
        
        # Initialize function implementations with correct parameters
        function_implementations = FunctionImplementations(
            cost_tracker=cost_tracker,
            memory_system=memory_system,
            constitution_manager=constitution_manager,
            model_manager=model_manager,
            config_manager=config,
            data_store=data_store,
            db_path="./auglab_bot.db"
        )
        
        # Register functions with the function manager
        function_manager.register_functions(function_implementations.get_all_functions())
        
        # Initialize response generator
        response_generator = ResponseGenerator(
            model_manager=model_manager,
            function_manager=function_manager,
            conversation_manager=conversation_manager,
            cost_tracker=cost_tracker,
            memory_system=memory_system,
            constitution_manager=constitution_manager,
            config_manager=config  # Changed from config to config_manager
        )
        
        # Set references for message handling
        response_generator.function_implementations = function_implementations
        
        # Create mock message
        mock_message = MockMessage(message_content, user_id, username, display_name)
        
        print(f"📨 Simulating message from {display_name} (@{username}, ID: {user_id})")
        print(f"💬 Message: \"{message_content}\"")
        print("-" * 60)
        
        # Generate response
        response = await response_generator.generate_response(mock_message, message_content)
        
        print(f"🤖 Dax Response:")
        print(f"{response}")
        print("-" * 60)
        print(f"✅ Simulation complete!")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function to run message simulation."""
    if len(sys.argv) < 2:
        print("Usage: python simulate_message.py \"<message>\" [user_id] [username] [display_name]")
        print("\nExamples:")
        print("  python simulate_message.py \"Hey dax, how are you?\"")

        print("  python simulate_message.py \"Can you tag vie and tell her about the project?\"")
        sys.exit(1)
    
    message_content = sys.argv[1]
    user_id = int(sys.argv[2]) if len(sys.argv) > 2 else 314159
    username = sys.argv[3] if len(sys.argv) > 3 else "m314x"
    display_name = sys.argv[4] if len(sys.argv) > 4 else "Test User"
    
    # Run the simulation
    asyncio.run(simulate_message(message_content, user_id, username, display_name))

if __name__ == "__main__":
    main() 