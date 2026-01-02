#!/usr/bin/env python3
"""
Dax Debug Terminal Client
========================

Interactive terminal interface for chatting with dax in debug mode.
Shows:
- Memories retrieved automatically
- Tools called by dax
- Tool results
- Full conversation with color-coded output
- **NEW: Autonomous mode for continuous strategic operations**

Usage: 
  python dax_debug_terminal.py                    # Interactive mode
  python dax_debug_terminal.py --autonomous       # Autonomous mode  
  python dax_debug_terminal.py --continuous       # Continuous autonomous mode
  python dax_debug_terminal.py --test             # Non-interactive test mode
  python dax_debug_terminal.py --prompt "message" # Non-interactive with single message
"""

import asyncio
import json
import sqlite3
import re
import sys
import os
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.rule import Rule
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to sys.path so we can import from the main bot file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main bot classes - NEW MODULAR STRUCTURE
from auglab_bot.config import ConfigManager
from auglab_bot.storage import CostTracker, MemorySystem
from auglab_bot.ai import ModelManager, FunctionCallManager, ResponseGenerator
from auglab_bot.bot.functions import FunctionImplementations
from auglab_bot_main import AugmentationLabBot

class DaxDebugTerminal:
    """Debug terminal interface for dax."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.console = Console()
        self.config_path = config_path
        self.config = ConfigManager(config_path)  # Use proper ConfigManager
        self.db_path = self.config.get('bot.database_path', './auglab_bot.db')
        self.budget = self.config.get('budget.total_budget', 5.0)
        
        # Initialize components
        self.memory_system = MemorySystem(self.db_path)
        self.cost_tracker = CostTracker(self.budget, self.db_path)
        
        # Initialize bot components for response generation
        self.bot = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Debug tracking
        self.debug_session = {
            'start_time': datetime.now(),
            'messages': [],
            'total_cost': 0.0,
            'tool_calls': 0,
            'memories_retrieved': 0,
            'autonomous_sessions': 0,
            'strategic_actions': 0
        }
        
        # Autonomous mode settings
        self.autonomous_mode_active = False
        self.autonomous_tool_calls_used = 0
        self.continuous_mode = False
        self.test_mode = False
            
    def _get_nested_config(self, key: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation."""
        return self.config.get(key, default)
            
    def _get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model configuration for cheap/expensive models."""
        if model_type == "cheap":
            return self.config.get_model_config('cheap')
        elif model_type == "expensive":
            return self.config.get_model_config('expensive')
        return {}

    async def initialize_bot(self):
        """Initialize the bot for response generation."""
        self.console.print("[yellow]Initializing dax debug session...[/yellow]")
        
        # Create a minimal bot-like object for generating responses
        class DebugBot:
            def __init__(self, config, db_path, budget):
                self.config = config
                self.db_path = db_path
                self.budget = budget
                
                # Initialize modular components
                self.memory_system = MemorySystem(db_path)
                self.cost_tracker = CostTracker(budget, db_path)
                self.model_manager = ModelManager(config)
                self.function_manager = FunctionCallManager()
                
                # Import additional components needed by ResponseGenerator
                from auglab_bot.management import ConstitutionManager
                from auglab_bot.ai import ConversationManager
                
                self.constitution_manager = ConstitutionManager(db_path)
                self.conversation_manager = ConversationManager()
                
                self.response_generator = ResponseGenerator(
                    model_manager=self.model_manager,
                    function_manager=self.function_manager,
                    conversation_manager=self.conversation_manager,
                    cost_tracker=self.cost_tracker,
                    memory_system=self.memory_system,
                    constitution_manager=self.constitution_manager,
                    config_manager=config
                )
                
                # Model configurations
                self.cheap_model_config = config.get_model_config('cheap')
                self.expensive_model_config = config.get_model_config('expensive')
                
                # Import additional components needed by FunctionImplementations
                from auglab_bot.storage import ChannelDataStore
                
                self.data_store = ChannelDataStore(db_path)
                
                # Initialize function implementations with correct parameters
                self.function_implementations = FunctionImplementations(
                    cost_tracker=self.cost_tracker,
                    memory_system=self.memory_system,
                    constitution_manager=self.constitution_manager,
                    model_manager=self.model_manager,
                    config_manager=config,
                    data_store=self.data_store,
                    db_path=db_path
                )
                
                # Register all functions with the function manager
                functions = self.function_implementations.get_all_functions()
                for func_name, func_impl in functions.items():
                    self.function_manager.register_function(func_name, func_impl)
                
                # Tokenizer for cost estimation
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                
                # Autonomous mode tracking
                self.autonomous_mode_active = False
                self.autonomous_tool_calls_used = 0
                
            def _get_model_config(self, model_type: str) -> Dict[str, Any]:
                """Get model configuration."""
                if model_type == "cheap":
                    return self.config.get_model_config('cheap')
                elif model_type == "expensive":
                    return self.config.get_model_config('expensive')
                return {'name': 'haiku', 'model': 'claude-3-haiku-20240307', 'provider': 'anthropic'}
                
        self.bot = DebugBot(self.config, self.db_path, self.budget)
        
        self.console.print("[green]✅ Debug session initialized![/green]")
        
    def display_welcome(self, mode: str = "interactive"):
        """Display welcome message with session info."""
        mode_info = {
            "interactive": "Interactive chat mode - Type messages to chat with dax",
            "autonomous": "Autonomous mode - Dax will run strategic intelligence operations",
            "continuous": "Continuous autonomous mode - Dax will run until budget exhausted",
            "test": "Test mode - Run non-interactive tests and verification",
            "prompt": "Non-interactive mode - Process single message and exit"
        }
        
        welcome_panel = Panel.fit(
            f"""[bold cyan]🤖 DAX DEBUG TERMINAL[/bold cyan]
            
[yellow]Mode:[/yellow] {mode.upper()}
[yellow]Session ID:[/yellow] {self.session_id}
[yellow]Database:[/yellow] {self.db_path}
[yellow]Budget:[/yellow] ${self.cost_tracker.get_remaining_budget():.2f}
[yellow]Models:[/yellow] {self.bot.cheap_model_config.get('name', 'haiku')} / {self.bot.expensive_model_config.get('name', 'sonnet')}

[dim]{mode_info.get(mode, 'Unknown mode')}
{'Type Ctrl+C to stop autonomous mode.' if mode != 'interactive' else "Type 'quit' or 'exit' to end the session."}[/dim]""",
            title="DEBUG MODE",
            border_style="cyan"
        )
        
        self.console.print(welcome_panel)

    async def run_test_mode(self):
        """Non-interactive test mode to verify everything works."""
        self.test_mode = True
        self.console.print("[yellow]🧪 Running non-interactive tests...[/yellow]")
        
        # Test 1: API Connection
        self.console.print("[blue]1. Testing API connection...[/blue]")
        try:
            # Create mock message for test
            class MockMessage:
                def __init__(self):
                    self.content = "Just say 'API test successful'"
                    self.author = MockUser(12345)
                    self.channel = MockChannel()
                    self.guild = None
                    
            class MockUser:
                def __init__(self, user_id: int):
                    self.id = user_id
                    self.display_name = "Test User"
                    
            class MockChannel:
                def __init__(self):
                    self.id = 12345
                    self.name = "test-channel"
                    
            message = MockMessage()
            
            # Use new modular system for test
            test_response = await self.bot.response_generator.generate_response(
                message=message,
                content="Just say 'API test successful'",
                use_expensive=False
            )
            self.console.print(f"[green]✅ API Connection: {test_response}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ API Connection failed: {e}[/red]")
            return False
            
        # Test 2: Memory System
        self.console.print("[blue]2. Testing memory system...[/blue]")
        try:
            memories = self.memory_system.query_memories("test", limit=3)
            self.console.print(f"[green]✅ Memory system: Found {len(memories)} memories[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Memory system failed: {e}[/red]")
            return False
            
        # Test 3: Cost Tracker
        self.console.print("[blue]3. Testing cost tracker...[/blue]")
        try:
            budget = self.cost_tracker.get_remaining_budget()
            self.console.print(f"[green]✅ Cost tracker: ${budget:.2f} remaining[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Cost tracker failed: {e}[/red]")
            return False
            
        # Test 4: Autonomous Mode (Mock)
        self.console.print("[blue]4. Testing autonomous mode simulation...[/blue]")
        try:
            mock_user = type('MockUser', (), {'id': 12345, 'display_name': 'Test User'})()
            mock_channel = type('MockChannel', (), {'id': 67890, 'name': 'test-channel'})()
            
            # Create mock message
            mock_message = type('MockMessage', (), {
                'author': mock_user,
                'channel': mock_channel,
                'content': 'Hello, this is a test message',
                'mentions': [],
                'guild': None
            })()
            
            # Test autonomous thinking prompt generation
            system_prompt = self.bot.config.get('system_prompt.autonomous_prompt', 'Default autonomous prompt')
            self.console.print(f"[green]✅ Autonomous mode: Ready with system prompt[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Autonomous mode test failed: {e}[/red]")
            return False
        
        # Test 5: Configuration Loading
        self.console.print("[blue]5. Testing configuration...[/blue]")
        try:
            anthropic_key = self.config.get('api_keys.anthropic')
            openai_key = self.config.get('api_keys.openai')
            
            if anthropic_key and openai_key:
                self.console.print(f"[green]✅ Configuration: API keys loaded[/green]")
            else:
                self.console.print(f"[red]❌ Configuration: Missing API keys[/red]")
                return False
                
        except Exception as e:
            self.console.print(f"[red]❌ Configuration test failed: {e}[/red]")
            return False
            
        self.console.print("[green]🎉 All tests passed! Debug terminal is ready.[/green]")
        return True

    async def run_prompt_mode(self, prompt: str):
        """Non-interactive mode - process single message and exit."""
        self.console.print(f"[yellow]🤖 Processing prompt: {prompt}[/yellow]")
        
        # Track message
        self.debug_session['messages'].append({
            'user': prompt,
            'timestamp': datetime.now()
        })
        
        # Show processing indicator
        with self.console.status("[yellow]Processing message...[/yellow]") as status:
            debug_info = await self.generate_debug_response(prompt)
            
        # Update session stats
        self.debug_session['total_cost'] += debug_info['cost']
        self.debug_session['tool_calls'] += len(debug_info['function_calls'])
        self.debug_session['memories_retrieved'] += len(debug_info['memories_retrieved'])
        
        # Display debug information
        self.console.print()
        
        # Show memories if any were retrieved
        if debug_info['memories_retrieved']:
            self.display_memories(debug_info['memories_retrieved'])
            
        # Show function calls if any were made
        if debug_info['function_calls']:
            self.display_function_calls(debug_info['function_calls'])
            
        # Show function results if any
        if debug_info['function_results']:
            self.display_function_results(debug_info['function_results'])
            
        # Show the response
        self.display_response(debug_info['response'])
        
        # Show cost info
        self.display_cost_info(debug_info['cost'], debug_info['model_used'])
        
        self.console.print(f"[green]✅ Prompt processed successfully![/green]")

    def display_autonomous_welcome(self):
        """Display autonomous mode specific welcome."""
        autonomous_panel = Panel.fit(
            f"""[bold yellow]🤖 AUTONOMOUS MODE ACTIVATED[/bold yellow]
            
[green]Strategic Intelligence Operations[/green]
Dax will now run continuous strategic operations including:
• Memory queries for strategic intelligence
• Research on viral content and project strategies  
• Analysis of member connections and opportunities
• Proactive strategic planning and execution

[yellow]Budget Requirement:[/yellow] ${self._get_nested_config('autonomous.budget_requirement', 1.0)} minimum
[yellow]Tool Calls per Session:[/yellow] {self._get_nested_config('autonomous.min_tool_calls', 15)}-{self._get_nested_config('autonomous.max_tool_calls', 25)}
[yellow]Current Budget:[/yellow] ${self.cost_tracker.get_remaining_budget():.2f}

[dim]Watch the real-time strategic operations below...[/dim]""",
            title="AUTONOMOUS OPERATIONS",
            border_style="yellow"
        )
        
        self.console.print(autonomous_panel)

    def display_memories(self, memories: List[Dict[str, Any]]):
        """Display retrieved memories in a formatted table."""
        if not memories:
            return
            
        table = Table(title="🧠 Retrieved Memories", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=12)
        table.add_column("Similarity", justify="right", style="cyan", width=10)
        table.add_column("Type", style="green", width=12)
        table.add_column("Content", style="white", width=60)
        
        for memory in memories:
            memory_id = memory.get('id', 'N/A')[:8] + "..."
            similarity = f"{memory.get('similarity', 0.0):.3f}"
            memory_type = memory.get('memory_type', 'unknown')
            content = memory.get('content', '')[:80] + "..." if len(memory.get('content', '')) > 80 else memory.get('content', '')
            
            table.add_row(memory_id, similarity, memory_type, content)
            
        self.console.print(table)

    def display_function_calls(self, function_calls: List[Dict[str, Any]]):
        """Display function calls in a formatted way."""
        if not function_calls:
            return
            
        self.console.print(Rule("[bold yellow]🔧 Function Calls[/bold yellow]"))
        
        for i, call in enumerate(function_calls, 1):
            func_name = call.get('function', 'unknown')
            params = call.get('parameters', {})
            
            # Color code different function types
            if func_name == 'think':
                color = "blue"
            elif func_name.startswith('query_'):
                color = "cyan"
            elif func_name.startswith('store_'):
                color = "green"
            elif func_name.startswith('edit_') or func_name.startswith('delete_'):
                color = "yellow"
            elif func_name == 'google_search':
                color = "magenta"
            elif func_name == 'trigger_autonomous_mode':
                color = "red"
            elif func_name == 'stop_autonomous_mode':
                color = "red"
            else:
                color = "white"
                
            self.console.print(f"[{color}]{i}. {func_name}[/{color}]")
            
            # Display parameters
            for key, value in params.items():
                if key == 'thought' and len(str(value)) > 100:
                    # Truncate long thoughts
                    value = str(value)[:100] + "..."
                elif key == 'new_content' and len(str(value)) > 80:
                    # Truncate long content
                    value = str(value)[:80] + "..."
                elif key == 'search_query' and len(str(value)) > 60:
                    # Truncate long search queries
                    value = str(value)[:60] + "..."
                    
                self.console.print(f"  [dim]{key}:[/dim] {value}")

    def display_function_results(self, results: str):
        """Display function call results."""
        if not results:
            return
            
        self.console.print(Rule("[bold green]✅ Function Results[/bold green]"))
        
        # Parse the results and display them nicely
        lines = results.split('\n')
        for line in lines:
            if line.startswith('✅'):
                self.console.print(f"[green]{line}[/green]")
            elif line.startswith('❌'):
                self.console.print(f"[red]{line}[/red]")
            elif line.startswith('🧠'):
                self.console.print(f"[cyan]{line}[/cyan]")
            elif line.startswith('💾'):
                self.console.print(f"[green]{line}[/green]")
            elif line.startswith('🔍'):
                self.console.print(f"[magenta]{line}[/magenta]")
            elif line.startswith('💭'):
                self.console.print(f"[blue]{line}[/blue]")
            else:
                self.console.print(line)

    def display_response(self, response: str):
        """Display the bot's response."""
        response_panel = Panel(
            response,
            title="[bold green]🤖 Dax Response[/bold green]",
            border_style="green"
        )
        self.console.print(response_panel)

    def display_cost_info(self, cost: float, model_used: str):
        """Display cost information."""
        remaining = self.cost_tracker.get_remaining_budget()
        percentage_used = ((self.budget - remaining) / self.budget) * 100
        
        cost_text = f"[yellow]Cost:[/yellow] ${cost:.4f} | [yellow]Model:[/yellow] {model_used} | [yellow]Remaining:[/yellow] ${remaining:.2f} ({percentage_used:.1f}% used)"
        self.console.print(cost_text)

    def display_autonomous_progress(self, current_calls: int, min_calls: int, max_calls: int):
        """Display autonomous mode progress."""
        progress_panel = Panel.fit(
            f"""[bold cyan]🤖 AUTONOMOUS PROGRESS[/bold cyan]
            
[yellow]Tool Calls Used:[/yellow] {current_calls}/{max_calls}
[yellow]Minimum Required:[/yellow] {min_calls}
[yellow]Budget Used:[/yellow] ${self.budget - self.cost_tracker.get_remaining_budget():.2f}
[yellow]Budget Remaining:[/yellow] ${self.cost_tracker.get_remaining_budget():.2f}
[yellow]Sessions Run:[/yellow] {self.debug_session['autonomous_sessions']}""",
            title="AUTONOMOUS STATUS",
            border_style="cyan"
        )
        
        self.console.print(progress_panel)

    async def generate_debug_response(self, user_input: str, user_id: int = 12345) -> Dict[str, Any]:
        """Generate a response with full debug information."""
        debug_info = {
            'memories_retrieved': [],
            'function_calls': [],
            'function_results': '',
            'response': '',
            'cost': 0.0,
            'model_used': 'haiku',
            'system_message': ''
        }
        
        # Create a mock message object for compatibility
        class MockMessage:
            def __init__(self, content: str, user_id: int):
                self.content = content
                self.author = MockUser(user_id)
                self.channel = MockChannel()
                self.guild = None
                
        class MockUser:
            def __init__(self, user_id: int):
                self.id = user_id
                self.display_name = "Debug User"
                self.name = "Debug User"
                
            def get(self, key, default=None):
                """Mock get method for compatibility."""
                if key == 'display_name':
                    return self.display_name
                elif key == 'id':
                    return self.id
                elif key == 'name':
                    return self.name
                return default
                
        class MockChannel:
            def __init__(self):
                self.id = 12345
                self.name = "debug-terminal"
                
        message = MockMessage(user_input, user_id)
        
        # Step 1: Retrieve relevant memories automatically
        if len(user_input.strip()) > 3:
            memories = self.memory_system.query_memories(
                user_input,
                user_id=user_id,
                limit=5
            )
            # Filter for high relevance
            high_relevance_memories = [m for m in memories if m.get('similarity', 0) >= 0.3]
            debug_info['memories_retrieved'] = high_relevance_memories
            
        # Step 2: Build system message with memories
        system_parts = [self._get_system_prompt()]
        
        if debug_info['memories_retrieved']:
            system_parts.append("\n## Relevant Memories:")
            for memory in debug_info['memories_retrieved']:
                similarity = memory.get('similarity', 0.0)
                timestamp = memory['timestamp'][:19]
                content_preview = memory['content'][:150]
                memory_type = memory['memory_type']
                system_parts.append(f"- [{similarity:.2f}] {timestamp} ({memory_type}): {content_preview}...")
                
        debug_info['system_message'] = "\n".join(system_parts)
        
        # Step 3: Generate response
        try:
            model_config = self.bot.cheap_model_config
            model_name = model_config.get('name', 'haiku')
            
            # Calculate tokens
            system_tokens = len(self.bot.tokenizer.encode(debug_info['system_message']))
            user_tokens = len(self.bot.tokenizer.encode(user_input))
            total_input_tokens = system_tokens + user_tokens
            
            max_tokens = 800
            
            if not self.cost_tracker.can_afford(model_name, total_input_tokens + max_tokens):
                debug_info['response'] = "⚠️ Insufficient budget for this request."
                return debug_info
                
            # Generate response using the model manager directly to capture function calls
            conversation_context = []
            if debug_info['memories_retrieved']:
                conversation_context = [{
                    'role': 'user',
                    'content': f"Context from memories: {str(debug_info['memories_retrieved'][:3])}"
                }]
            
            conversation_context.append({
                'role': 'user',
                'content': user_input
            })
            
            response_text = await self.bot.model_manager.call_model_api(
                model_config,
                debug_info['system_message'],
                conversation_context=conversation_context,
                max_tokens=max_tokens
            )
            
            # Parse and execute function calls
            function_calls = self.bot.function_manager.parse_function_calls(response_text)
            debug_info['function_calls'] = function_calls
            
            if function_calls:
                function_results = await self.bot.function_manager.execute_function_calls(function_calls)
                debug_info['function_results'] = str(function_results)
            
            # Calculate cost
            model_name = model_config.get('name', 'haiku')
            output_tokens = len(self.bot.tokenizer.encode(response_text)) if response_text else 0
            cost = self.cost_tracker.calculate_cost(model_name, total_input_tokens, output_tokens)
            
            # Cost is already recorded by ResponseGenerator, just track it locally
            self.debug_session['total_cost'] += cost
            
            debug_info['response'] = self._clean_xml_tags(response_text)
            debug_info['cost'] = cost
            debug_info['model_used'] = model_name
                
        except Exception as e:
            debug_info['response'] = f"Error generating response: {str(e)}"
            
        return debug_info

    async def generate_autonomous_response(self) -> Dict[str, Any]:
        """Generate autonomous strategic response."""
        # Check budget requirement
        budget_requirement = self._get_nested_config('autonomous.budget_requirement', 1.0)
        if self.cost_tracker.get_remaining_budget() < budget_requirement:
            return {
                'response': f"⚠️ Insufficient budget for autonomous mode (${self.cost_tracker.get_remaining_budget():.2f} < ${budget_requirement})",
                'cost': 0.0,
                'function_calls': [],
                'function_results': '',
                'can_continue': False
            }
            
        # Activate autonomous mode
        self.autonomous_mode_active = True
        self.autonomous_tool_calls_used = 0
        
        # Get current context for autonomous thinking
        current_context = await self._gather_autonomous_context()
        
        # Generate autonomous thinking prompt
        base_autonomous_prompt = self._get_nested_config('system_prompt.autonomous_prompt', '')
        if isinstance(base_autonomous_prompt, list):
            base_autonomous_prompt = '\n'.join(base_autonomous_prompt)
            
        system_prompt = self._get_system_prompt()
        
        thinking_prompt = f"""{system_prompt}

{base_autonomous_prompt}

**CURRENT INTELLIGENCE:**
{current_context}

**AVAILABLE TOOLS:**
- <think>strategic analysis</think> - Internal reasoning (hidden from users)
- <query_memory>search terms</query_memory> - Search your intelligence database
- <store_observation user_id="12345" channel_id="12345" tags="strategy,intel">observation</store_observation> - Store strategic intelligence
- <google_search>strategic research query</google_search> - Gather external intelligence
- <get_current_time></get_current_time> - Time-based strategic context
- <stop_autonomous_mode reason="completed strategic session">summary</stop_autonomous_mode> - ONLY after 15+ tool calls

**AUTONOMOUS SESSION GOAL:**
Execute strategic intelligence operations to advance Aug Lab success metrics. Use 15-25 tool calls systematically."""

        debug_info = {
            'response': '',
            'cost': 0.0,
            'model_used': 'sonnet',
            'function_calls': [],
            'function_results': '',
            'can_continue': True
        }
        
        try:
            # Use expensive model for autonomous thinking
            model_config = self.bot.expensive_model_config
            model_name = model_config.get('name', 'sonnet')
            
            # Calculate tokens
            system_tokens = len(self.bot.tokenizer.encode(thinking_prompt))
            max_tokens = 1200
            
            if not self.cost_tracker.can_afford(model_name, system_tokens + max_tokens):
                debug_info['response'] = "⚠️ Insufficient budget for autonomous mode."
                debug_info['can_continue'] = False
                return debug_info
                
            # Create mock message for autonomous mode
            class MockMessage:
                def __init__(self):
                    self.content = "Begin autonomous strategic intelligence operations now."
                    self.author = MockUser(12345)
                    self.channel = MockChannel()
                    self.guild = None
                    
            class MockUser:
                def __init__(self, user_id: int):
                    self.id = user_id
                    self.display_name = "Autonomous System"
                    
            class MockChannel:
                def __init__(self):
                    self.id = 12345
                    self.name = "autonomous-mode"
                    
            message = MockMessage()
            
            # Generate response using the model manager directly to capture function calls
            response_text = await self.bot.model_manager.call_model_api(
                model_config,
                thinking_prompt,
                max_tokens=max_tokens
            )
            
            # Parse and execute function calls
            function_calls = self.bot.function_manager.parse_function_calls(response_text)
            debug_info['function_calls'] = function_calls
            
            if function_calls:
                function_results = await self.bot.function_manager.execute_function_calls(function_calls)
                debug_info['function_results'] = str(function_results)
            
            # Calculate cost
            output_tokens = len(self.bot.tokenizer.encode(response_text)) if response_text else 0
            cost = self.cost_tracker.calculate_cost(model_name, system_tokens, output_tokens)
            
            # Cost is already recorded by ResponseGenerator, just track it locally
            self.debug_session['total_cost'] += cost
            
            debug_info['response'] = self._clean_xml_tags(response_text)
            debug_info['cost'] = cost
            debug_info['model_used'] = model_name
            
            # Update tool call count for autonomous mode
            self.autonomous_tool_calls_used += len(function_calls)
            
            # Check for stop condition
            stop_calls = [call for call in function_calls if call.get('function') == 'stop_autonomous_mode']
            if stop_calls:
                self.autonomous_mode_active = False
                debug_info['can_continue'] = False
            else:
                # Check if we should continue based on budget and tool calls
                max_calls = self._get_nested_config('autonomous.max_tool_calls', 25)
                if (self.autonomous_tool_calls_used >= max_calls or 
                    self.cost_tracker.get_remaining_budget() < budget_requirement):
                    debug_info['can_continue'] = False
                    self.autonomous_mode_active = False
            
            # Clean XML tags from response
            debug_info['response'] = self._clean_xml_tags(response_text)
                
        except Exception as e:
            debug_info['response'] = f"Error in autonomous mode: {str(e)}"
            debug_info['can_continue'] = False
            self.autonomous_mode_active = False
            
        return debug_info

    async def _gather_autonomous_context(self) -> str:
        """Gather current context for autonomous thinking."""
        context_parts = []
        
        # Get recent memories
        try:
            recent_memories = self.memory_system.query_memories(
                "project social media member collaboration strategy",
                limit=5
            )
            if recent_memories:
                context_parts.append("Recent Strategic Intelligence:")
                for memory in recent_memories:
                    content = memory['content'][:100] + "..." if len(memory['content']) > 100 else memory['content']
                    context_parts.append(f"- [{memory.get('similarity', 0.0):.2f}] {content}")
        except:
            pass
            
        # Get budget info
        remaining_budget = self.cost_tracker.get_remaining_budget()
        context_parts.append(f"Budget Status: ${remaining_budget:.2f} remaining")
        
        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context_parts.append(f"Current Time: {current_time}")
        
        # Add session stats
        context_parts.append(f"Session Stats: {self.debug_session['autonomous_sessions']} autonomous sessions run")
        
        return "\n".join(context_parts)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for debug mode."""
        main_prompt = self._get_nested_config('system_prompt.main_prompt', '')
        if isinstance(main_prompt, list):
            main_prompt = '\n'.join(main_prompt)
            
        return f"""{main_prompt}

## Debug Mode Context
You are running in debug mode. Be strategic and show your reasoning process.

## Available Functions
- <think>strategic analysis</think> - Show your thought process
- <query_memory>search terms</query_memory> - Search memory system
- <store_observation user_id="12345" channel_id="12345" tags="debug,strategy">observation</store_observation> - Store information
- <edit_memory memory_id="memory_id" reason="correction reason">updated content</edit_memory> - Edit existing memory
- <delete_memory memory_id="memory_id" reason="deletion reason">unused</delete_memory> - Delete memory
- <google_search>search query</google_search> - Web search for strategic intelligence
- <get_current_time></get_current_time> - Get current time
- <stop_autonomous_mode reason="session complete">summary</stop_autonomous_mode> - Stop autonomous mode"""

    def _parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse function calls from AI response."""
        function_calls = []
        
        patterns = {
            'think': r'<think>(.*?)</think>',
            'query_memory': r'<query_memory>(.*?)</query_memory>',
            'store_observation': r'<store_observation user_id="(\d+)" channel_id="(\d+)" tags="([^"]*)">(.*?)</store_observation>',
            'edit_memory': r'<edit_memory memory_id="([^"]*)" reason="([^"]*)">(.*?)</edit_memory>',
            'delete_memory': r'<delete_memory memory_id="([^"]*)" reason="([^"]*)">(.*?)</delete_memory>',
            'google_search': r'<google_search>(.*?)</google_search>',
            'get_current_time': r'<get_current_time>(.*?)</get_current_time>',
            'stop_autonomous_mode': r'<stop_autonomous_mode reason="([^"]*)">(.*?)</stop_autonomous_mode>'
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
                elif func_name == 'google_search':
                    function_calls.append({
                        'function': 'google_search',
                        'parameters': {'search_query': match.strip()}
                    })
                elif func_name == 'get_current_time':
                    function_calls.append({
                        'function': 'get_current_time',
                        'parameters': {}
                    })
                elif func_name == 'stop_autonomous_mode':
                    reason, summary = match
                    function_calls.append({
                        'function': 'stop_autonomous_mode',
                        'parameters': {
                            'reason': reason.strip(),
                            'summary': summary.strip()
                        }
                    })
                    
        return function_calls

    async def _execute_function_calls(self, function_calls: List[Dict[str, Any]]) -> str:
        """Execute function calls and return results."""
        results = []
        
        for call in function_calls:
            func_name = call.get('function')
            params = call.get('parameters', {})
            
            # Track tool calls for autonomous mode
            if self.autonomous_mode_active:
                self.autonomous_tool_calls_used += 1
                
            try:
                if func_name == 'think':
                    results.append(f"💭 **Thought**: {params.get('thought', '')}")
                    
                elif func_name == 'query_memory':
                    query = params.get('query', '')
                    memories = self.memory_system.query_memories(query, limit=10)
                    results.append(f"🧠 **Memory Query**: '{query}' → Found {len(memories)} memories")
                    for i, memory in enumerate(memories[:3], 1):
                        content = memory['content'][:100] + "..." if len(memory['content']) > 100 else memory['content']
                        results.append(f"  {i}. [{memory.get('similarity', 0.0):.2f}] {content}")
                        
                elif func_name == 'store_observation':
                    observation = params.get('observation', '')
                    user_id = params.get('user_id', 12345)
                    channel_id = params.get('channel_id', 12345)
                    tags = params.get('tags', [])
                    
                    memory_id = self.memory_system.store_memory(
                        observation, user_id, channel_id, 
                        memory_type='observation', tags=tags
                    )
                    results.append(f"💾 **Stored**: {memory_id[:8]}... - {observation[:50]}...")
                    
                elif func_name == 'edit_memory':
                    memory_id = params.get('memory_id', '')
                    reason = params.get('reason', '')
                    new_content = params.get('new_content', '')
                    
                    success = self.memory_system.edit_memory(memory_id, new_content, reason)
                    results.append(f"✏️ **Edit**: {memory_id[:8]}... - {'Success' if success else 'Failed'}")
                    
                elif func_name == 'delete_memory':
                    memory_id = params.get('memory_id', '')
                    reason = params.get('reason', '')
                    
                    success = self.memory_system.delete_memory(memory_id, reason)
                    results.append(f"🗑️ **Delete**: {memory_id[:8]}... - {'Success' if success else 'Failed'}")
                    
                elif func_name == 'google_search':
                    search_query = params.get('search_query', '')
                    # Mock google search for debug mode
                    results.append(f"🔍 **Google Search**: '{search_query}' → [Mock search results - integration required]")
                    
                elif func_name == 'get_current_time':
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    results.append(f"🕐 **Current Time**: {current_time}")
                    
                elif func_name == 'stop_autonomous_mode':
                    reason = params.get('reason', '')
                    summary = params.get('summary', '')
                    self.autonomous_mode_active = False
                    results.append(f"🛑 **Autonomous Mode Stopped**: {reason}")
                    if summary:
                        results.append(f"  Summary: {summary}")
                    
                else:
                    results.append(f"❌ **{func_name}**: Function not implemented in debug mode")
                    
            except Exception as e:
                results.append(f"❌ **{func_name}**: Error - {str(e)}")
                
        return "\n".join(results)

    def _clean_xml_tags(self, text: str) -> str:
        """Remove XML function call tags from response."""
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'<query_memory>.*?</query_memory>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<store_observation.*?>.*?</store_observation>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<edit_memory.*?>.*?</edit_memory>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<delete_memory.*?>.*?</delete_memory>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<google_search>.*?</google_search>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<get_current_time>.*?</get_current_time>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<stop_autonomous_mode.*?>.*?</stop_autonomous_mode>', '', cleaned, flags=re.DOTALL)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned

    def display_session_summary(self):
        """Display session summary."""
        duration = datetime.now() - self.debug_session['start_time']
        
        summary_panel = Panel.fit(
            f"""[bold cyan]🎯 DEBUG SESSION SUMMARY[/bold cyan]
            
[yellow]Duration:[/yellow] {duration}
[yellow]Messages:[/yellow] {len(self.debug_session['messages'])}
[yellow]Total Cost:[/yellow] ${self.debug_session['total_cost']:.4f}
[yellow]Tool Calls:[/yellow] {self.debug_session['tool_calls']}
[yellow]Memories Retrieved:[/yellow] {self.debug_session['memories_retrieved']}
[yellow]Autonomous Sessions:[/yellow] {self.debug_session['autonomous_sessions']}
[yellow]Strategic Actions:[/yellow] {self.debug_session['strategic_actions']}
[yellow]Budget Remaining:[/yellow] ${self.cost_tracker.get_remaining_budget():.2f}""",
            title="SESSION COMPLETE",
            border_style="green"
        )
        
        self.console.print(summary_panel)

    async def run_interactive(self):
        """Run in interactive chat mode."""
        self.display_welcome("interactive")
        
        try:
            while True:
                # Get user input
                user_input = self.console.input("\n[bold cyan]You:[/bold cyan] ")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                    continue
                    
                # Track message
                self.debug_session['messages'].append({
                    'user': user_input,
                    'timestamp': datetime.now()
                })
                
                # Show processing indicator
                with self.console.status("[yellow]Processing...[/yellow]") as status:
                    debug_info = await self.generate_debug_response(user_input)
                    
                # Update session stats
                self.debug_session['total_cost'] += debug_info['cost']
                self.debug_session['tool_calls'] += len(debug_info['function_calls'])
                self.debug_session['memories_retrieved'] += len(debug_info['memories_retrieved'])
                
                # Display debug information
                self.console.print()
                
                # Show memories if any were retrieved
                if debug_info['memories_retrieved']:
                    self.display_memories(debug_info['memories_retrieved'])
                    
                # Show function calls if any were made
                if debug_info['function_calls']:
                    self.display_function_calls(debug_info['function_calls'])
                    
                # Show function results if any
                if debug_info['function_results']:
                    self.display_function_results(debug_info['function_results'])
                    
                # Show the response
                self.display_response(debug_info['response'])
                
                # Show cost info
                self.display_cost_info(debug_info['cost'], debug_info['model_used'])
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Session interrupted by user.[/yellow]")

    async def _handle_command(self, command: str):
        """Handle slash commands."""
        command = command.lower().strip()
        
        if command in ['/help', '/h']:
            self._show_help()
        elif command in ['/status', '/s']:
            self._show_status()
        elif command in ['/budget', '/b']:
            self._show_budget()
        elif command in ['/memory', '/m']:
            await self._show_memory()
        elif command in ['/config', '/c']:
            self._show_config()
        elif command in ['/autonomous', '/auto', '/a']:
            await self._enter_autonomous_mode()
        elif command in ['/clear', '/cl']:
            self._clear_screen()
        elif command in ['/test', '/t']:
            await self._run_tests()
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[yellow]Type '/help' for available commands[/yellow]")

    def _show_help(self):
        """Show help menu."""
        help_panel = Panel.fit(
            """[bold yellow]🛠️ DEBUG TERMINAL COMMANDS[/bold yellow]
            
[bold cyan]Chat Commands:[/bold cyan]
  [yellow]/help, /h[/yellow]          - Show this help menu
  [yellow]/status, /s[/yellow]        - Show current session status
  [yellow]/budget, /b[/yellow]        - Show budget information
  [yellow]/memory, /m[/yellow]        - Show memory statistics
  [yellow]/config, /c[/yellow]        - Show configuration
  [yellow]/autonomous, /auto, /a[/yellow] - Enter autonomous mode
  [yellow]/clear, /cl[/yellow]        - Clear the screen
  [yellow]/test, /t[/yellow]          - Run system tests
  [yellow]quit, exit, q[/yellow]      - Exit the terminal

[bold cyan]Regular Chat:[/bold cyan]
  Just type your message and press Enter to chat with dax
  Example: "Hello, how are you doing today?"

[bold cyan]Autonomous Mode:[/bold cyan]
  In autonomous mode, dax will run strategic intelligence operations
  including memory queries, research, and proactive planning.
  Use '/autonomous' to enter this mode.

[bold cyan]Function Calls:[/bold cyan]
  Watch for function calls in responses - they appear as XML tags
  like <query_memory>search terms</query_memory>
  
[dim]Tip: All interactions are logged and cost-tracked![/dim]""",
            title="HELP MENU",
            border_style="cyan"
        )
        
        self.console.print(help_panel)

    def _show_status(self):
        """Show current session status."""
        duration = datetime.now() - self.debug_session['start_time']
        
        status_panel = Panel.fit(
            f"""[bold cyan]📊 SESSION STATUS[/bold cyan]
            
[yellow]Duration:[/yellow] {duration}
[yellow]Messages:[/yellow] {len(self.debug_session['messages'])}
[yellow]Total Cost:[/yellow] ${self.debug_session['total_cost']:.4f}
[yellow]Tool Calls:[/yellow] {self.debug_session['tool_calls']}
[yellow]Memories Retrieved:[/yellow] {self.debug_session['memories_retrieved']}
[yellow]Autonomous Sessions:[/yellow] {self.debug_session['autonomous_sessions']}
[yellow]Strategic Actions:[/yellow] {self.debug_session['strategic_actions']}
[yellow]Budget Remaining:[/yellow] ${self.cost_tracker.get_remaining_budget():.2f}
[yellow]Autonomous Mode:[/yellow] {'Active' if self.autonomous_mode_active else 'Inactive'}""",
            title="SESSION STATUS",
            border_style="blue"
        )
        
        self.console.print(status_panel)

    def _show_budget(self):
        """Show budget information."""
        try:
            report = self.cost_tracker.get_spending_report()
            percentage_used = (report['total_spent'] / report['budget']) * 100
            
            budget_panel = Panel.fit(
                f"""[bold green]💰 BUDGET INFORMATION[/bold green]
                
[yellow]Total Budget:[/yellow] ${report['budget']:.2f}
[yellow]Spent:[/yellow] ${report['total_spent']:.4f}
[yellow]Remaining:[/yellow] ${report['remaining']:.2f}
[yellow]Percentage Used:[/yellow] {percentage_used:.1f}%
[yellow]Can Afford Expensive:[/yellow] {'Yes' if self.cost_tracker.can_afford('sonnet', 2000) else 'No'}
[yellow]Can Afford Cheap:[/yellow] {'Yes' if self.cost_tracker.can_afford('haiku', 1000) else 'No'}

[bold cyan]Spending by Model:[/bold cyan]
{self._format_spending_by_model(report.get('by_model', {}))}""",
                title="BUDGET STATUS",
                border_style="green"
            )
            
            self.console.print(budget_panel)
        except Exception as e:
            self.console.print(f"[red]Error getting budget info: {e}[/red]")

    def _format_spending_by_model(self, by_model: Dict[str, float]) -> str:
        """Format spending by model."""
        if not by_model:
            return "[dim]No spending data available[/dim]"
        
        lines = []
        for model, cost in by_model.items():
            lines.append(f"  {model}: ${cost:.4f}")
        
        return "\n".join(lines)

    async def _show_memory(self):
        """Show memory statistics."""
        try:
            # Get memory stats
            stats = self.memory_system.get_stats()
            
            memory_panel = Panel.fit(
                f"""[bold magenta]🧠 MEMORY STATISTICS[/bold magenta]
                
[yellow]Total Memories:[/yellow] {stats.get('total_memories', 0)}
[yellow]Memory Types:[/yellow] {', '.join(stats.get('memory_types', []))}
[yellow]Most Active User:[/yellow] {stats.get('most_active_user', 'N/A')}
[yellow]Recent Activity:[/yellow] {stats.get('recent_activity', 'N/A')}
[yellow]Database Size:[/yellow] {stats.get('database_size', 'N/A')}

[bold cyan]Recent Memories:[/bold cyan]
{await self._format_recent_memories()}""",
                title="MEMORY SYSTEM",
                border_style="magenta"
            )
            
            self.console.print(memory_panel)
        except Exception as e:
            self.console.print(f"[red]Error getting memory info: {e}[/red]")

    async def _format_recent_memories(self) -> str:
        """Format recent memories."""
        try:
            memories = self.memory_system.query_memories("", limit=5)
            if not memories:
                return "[dim]No memories found[/dim]"
            
            lines = []
            for memory in memories:
                content = memory['content'][:60] + "..." if len(memory['content']) > 60 else memory['content']
                lines.append(f"  • {content}")
            
            return "\n".join(lines)
        except Exception as e:
            return f"[red]Error: {e}[/red]"

    def _show_config(self):
        """Show configuration information."""
        config_panel = Panel.fit(
            f"""[bold yellow]⚙️ CONFIGURATION[/bold yellow]
            
[yellow]Config Path:[/yellow] {self.config_path}
[yellow]Database Path:[/yellow] {self.db_path}
[yellow]Budget:[/yellow] ${self.budget}
[yellow]Cheap Model:[/yellow] {self.config.get_model_config('cheap').get('name', 'Unknown')}
[yellow]Expensive Model:[/yellow] {self.config.get_model_config('expensive').get('name', 'Unknown')}

[bold cyan]API Keys:[/bold cyan]
[yellow]Anthropic:[/yellow] {'✅ Configured' if self.config.get('api_keys.anthropic') else '❌ Missing'}
[yellow]OpenAI:[/yellow] {'✅ Configured' if self.config.get('api_keys.openai') else '❌ Missing'}
[yellow]OpenRouter:[/yellow] {'✅ Configured' if self.config.get('api_keys.openrouter') else '❌ Missing'}""",
            title="CONFIGURATION",
            border_style="yellow"
        )
        
        self.console.print(config_panel)

    async def _enter_autonomous_mode(self):
        """Enter autonomous mode."""
        if self.autonomous_mode_active:
            self.console.print("[yellow]Already in autonomous mode![/yellow]")
            return
        
        if self.cost_tracker.get_remaining_budget() < 1.0:
            self.console.print("[red]Insufficient budget for autonomous mode![/red]")
            return
        
        self.console.print("[yellow]Entering autonomous mode...[/yellow]")
        self.autonomous_mode_active = True
        
        # Run one autonomous session
        debug_info = await self.generate_autonomous_response()
        
        # Update session stats
        self.debug_session['total_cost'] += debug_info['cost']
        self.debug_session['tool_calls'] += len(debug_info['function_calls'])
        self.debug_session['autonomous_sessions'] += 1
        
        # Display results
        if debug_info['function_calls']:
            self.display_function_calls(debug_info['function_calls'])
        if debug_info['function_results']:
            self.display_function_results(debug_info['function_results'])
        if debug_info['response']:
            self.display_response(debug_info['response'])
        
        self.display_cost_info(debug_info['cost'], debug_info['model_used'])
        
        self.autonomous_mode_active = False
        self.console.print("[green]Autonomous session completed![/green]")

    def _clear_screen(self):
        """Clear the terminal screen."""
        self.console.clear()
        self.console.print("[green]Screen cleared![/green]")

    async def _run_tests(self):
        """Run system tests."""
        self.console.print("[yellow]Running system tests...[/yellow]")
        success = await self.run_test_mode()
        if success:
            self.console.print("[green]✅ All tests passed![/green]")
        else:
            self.console.print("[red]❌ Some tests failed![/red]")

    async def run_autonomous(self, continuous: bool = False):
        """Run in autonomous mode."""
        self.continuous_mode = continuous
        mode = "continuous" if continuous else "autonomous"
        
        self.display_welcome(mode)
        self.display_autonomous_welcome()
        
        try:
            session_count = 0
            while True:
                session_count += 1
                self.debug_session['autonomous_sessions'] = session_count
                
                self.console.print(f"\n[bold yellow]🚀 Starting Autonomous Session #{session_count}[/bold yellow]")
                
                # Show processing indicator
                with self.console.status("[yellow]Running strategic intelligence operations...[/yellow]") as status:
                    debug_info = await self.generate_autonomous_response()
                    
                # Update session stats
                self.debug_session['total_cost'] += debug_info['cost']
                self.debug_session['tool_calls'] += len(debug_info['function_calls'])
                self.debug_session['strategic_actions'] += len(debug_info['function_calls'])
                
                # Display autonomous progress
                min_calls = self._get_nested_config('autonomous.min_tool_calls', 15)
                max_calls = self._get_nested_config('autonomous.max_tool_calls', 25)
                self.display_autonomous_progress(self.autonomous_tool_calls_used, min_calls, max_calls)
                
                # Display debug information
                self.console.print()
                
                # Show function calls if any were made
                if debug_info['function_calls']:
                    self.display_function_calls(debug_info['function_calls'])
                    
                # Show function results if any
                if debug_info['function_results']:
                    self.display_function_results(debug_info['function_results'])
                    
                # Show the response
                if debug_info['response']:
                    self.display_response(debug_info['response'])
                
                # Show cost info
                self.display_cost_info(debug_info['cost'], debug_info['model_used'])
                
                # Check if we should continue
                if not debug_info['can_continue']:
                    break
                    
                # In single autonomous mode, stop after one session
                if not continuous:
                    break
                    
                # Brief pause between sessions in continuous mode
                if continuous:
                    await asyncio.sleep(2)
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Autonomous mode interrupted by user.[/yellow]")
            self.autonomous_mode_active = False

    async def run(self, mode: str = "interactive", prompt: str = None):
        """Run the debug terminal in specified mode."""
        await self.initialize_bot()
        
        if mode == "interactive":
            await self.run_interactive()
        elif mode == "autonomous":
            await self.run_autonomous(continuous=False)
        elif mode == "continuous":
            await self.run_autonomous(continuous=True)
        elif mode == "test":
            await self.run_test_mode()
        elif mode == "prompt":
            if not prompt:
                self.console.print("[red]Error: --prompt requires a message argument[/red]")
                return
            await self.run_prompt_mode(prompt)
        else:
            self.console.print(f"[red]Unknown mode: {mode}[/red]")
            return
            
        self.display_session_summary()

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Dax Debug Terminal')
    parser.add_argument('--autonomous', '-a', action='store_true', 
                       help='Run in autonomous mode (single session)')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Run in continuous autonomous mode')
    parser.add_argument('--test', action='store_true',
                       help='Run in non-interactive test mode')
    parser.add_argument('--prompt', type=str,
                       help='Process single message in non-interactive mode')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.prompt:
        mode = "prompt"
        prompt = args.prompt
    elif args.continuous:
        mode = "continuous"
        prompt = None
    elif args.autonomous:
        mode = "autonomous"
        prompt = None
    elif args.test:
        mode = "test"
        prompt = None
    else:
        mode = "interactive"
        prompt = None
    
    terminal = DaxDebugTerminal()
    await terminal.run(mode, prompt)

if __name__ == "__main__":
    asyncio.run(main()) 