#!/usr/bin/env python3
"""
Simulate @bot mentions from the command line
Test expensive model responses as if someone mentioned the bot in Discord
"""

import os
import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import argparse

# Import the bot components
from auglab_bot.config import ConfigManager
from auglab_bot.storage import CostTracker, MemorySystem, CostEntry
import anthropic
import openai
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BotMentionSimulator:
    """Simulate @bot mentions from command line using the same logic as the Discord bot."""
    
    def __init__(self):
        # Load configuration
        self.config = ConfigManager()
        
        # Initialize cost tracking and memory
        self.cost_tracker = CostTracker(self.config.get('budget.total_budget', 5.0), './auglab_bot.db')
        self.memory_system = MemorySystem('./auglab_bot.db')
        
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
        else:
            self.openrouter_client = None
        
        # Tokenizer for cost estimation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Model configs
        self.cheap_model_config = self.config.get_model_config('cheap')
        self.expensive_model_config = self.config.get_model_config('expensive')
        
        # Simulated user info
        self.user_id = 999999999  # CLI user ID
        self.channel_id = 888888888  # CLI channel ID
        self.username = "CLI_User"
        
    def get_system_prompt(self) -> str:
        """Get the system prompt from configuration."""
        main_prompt = self.config.get('system_prompt.main_prompt', '')
        
        # Handle both string and array formats for backward compatibility
        if isinstance(main_prompt, list):
            main_prompt = '\n'.join(main_prompt)
        
        return main_prompt
    
    def get_importance_criteria(self) -> str:
        """Get current importance criteria for message filtering."""
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
    
    async def assess_importance(self, content: str) -> Dict[str, Any]:
        """Use Haiku to assess if a message needs expensive model treatment."""
        try:
            importance_criteria = self.get_importance_criteria()
            
            assessment_prompt = f"""You are a message importance filter for the Aug Lab Discord bot. Your job is to decide if this message needs the expensive Sonnet model or if Haiku can handle it.

## CURRENT IMPORTANCE CRITERIA
{importance_criteria}

## MESSAGE TO ASSESS
Message: "{content}"
Author: {self.username}
Channel: #cli-simulation
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
                user_id=self.user_id,
                channel_id=self.channel_id
            ))
            
            print(f"🤖 Haiku Assessment: {response.content[0].text}")
            
            # Parse JSON response
            try:
                result = json.loads(response.content[0].text)
                return result
            except:
                # Fallback if JSON parsing fails
                return {'use_expensive_model': True, 'reasoning': 'Parse error, using expensive model'}
                
        except Exception as e:
            logger.error(f"Error in importance assessment: {e}")
            return {'use_expensive_model': False, 'reasoning': 'Error occurred'}
    
    async def _call_model_api(self, model_config: Dict, prompt: str, max_tokens: int) -> str:
        """Call the appropriate API based on model configuration."""
        provider = model_config.get('provider', 'anthropic')
        model = model_config.get('model')
        
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
            if not self.openrouter_client:
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
    
    async def generate_response(self, content: str, force_expensive: Optional[bool] = None) -> str:
        """Generate AI response using the same logic as the Discord bot."""
        try:
            # Get user history
            user_history = self.memory_system.get_user_interaction_history(self.user_id, 5)
            
            print(f"👤 User History: Found {len(user_history)} previous interactions")
            
            # Determine model selection
            if force_expensive is not None:
                use_expensive = force_expensive and self.cost_tracker.can_afford('sonnet', 2000)
            else:
                # Use importance assessment (same as Discord bot)
                print("🔍 Assessing message importance...")
                importance_assessment = await self.assess_importance(content)
                use_expensive = importance_assessment['use_expensive_model'] and self.cost_tracker.can_afford('sonnet', 2000)
                
                print(f"📊 Assessment Result:")
                print(f"  Strategic Value: {importance_assessment.get('strategic_value', 'N/A')}")
                print(f"  Technical Complexity: {importance_assessment.get('technical_complexity', 'N/A')}")
                print(f"  Social Connection: {importance_assessment.get('social_connection', 'N/A')}")
                print(f"  Project Help: {importance_assessment.get('project_help', 'N/A')}")
                print(f"  Use Expensive Model: {importance_assessment.get('use_expensive_model', False)}")
                print(f"  Reasoning: {importance_assessment.get('reasoning', 'N/A')}")
            
            # Select model configuration
            model_config = self.expensive_model_config if use_expensive else self.cheap_model_config
            model_name = model_config.get('name', 'haiku')
            
            print(f"🤖 Using Model: {model_name} ({model_config.get('provider', 'unknown')}/{model_config.get('model', 'unknown')})")
            
            # Build context (same as Discord bot)
            context_parts = [self.get_system_prompt()]
            
            # Add user history context
            if user_history:
                context_parts.append("**Recent conversation history:**")
                for entry in user_history[:3]:  # Last 3 interactions
                    context_parts.append(f"Previous: {entry['content'][:100]}...")
            
            context_parts.append(f"**Current message from {self.username}:** {content}")
            
            full_prompt = "\n".join(context_parts)
            input_tokens = len(self.tokenizer.encode(full_prompt))
            max_tokens = 1500 if use_expensive else 800
            
            if not self.cost_tracker.can_afford(model_name, input_tokens + max_tokens):
                return "⚠️ Insufficient budget for this request. Check budget with --budget flag."
            
            # Generate response
            print(f"💭 Generating response...")
            response_text = await self._call_model_api(model_config, full_prompt, max_tokens)
            
            # Calculate and record cost
            output_tokens = len(self.tokenizer.encode(response_text))
            cost = self.cost_tracker.calculate_cost(model_name, input_tokens, output_tokens)
            
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                action='cli_bot_mention',
                user_id=self.user_id,
                channel_id=self.channel_id
            ))
            
            # Store interaction in memory
            interaction_content = f"User: {content}\nBot: {response_text}"
            self.memory_system.store_memory(
                interaction_content,
                self.user_id,
                self.channel_id,
                memory_type='interaction',
                tags=['cli', 'bot_mention']
            )
            
            # Show cost info
            print(f"💰 Cost: ${cost:.4f} (Input: {input_tokens} tokens, Output: {output_tokens} tokens)")
            print(f"💳 Remaining Budget: ${self.cost_tracker.get_remaining_budget():.2f}")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"❌ Error: {str(e)}"
    
    def show_budget(self):
        """Show current budget status."""
        print(f"💰 Budget Status:")
        print(f"  Total: ${self.cost_tracker.budget:.2f}")
        print(f"  Remaining: ${self.cost_tracker.get_remaining_budget():.2f}")
        print(f"  Spent: ${self.cost_tracker.budget - self.cost_tracker.get_remaining_budget():.2f}")
        
        # Show recent usage
        try:
            import sqlite3
            conn = sqlite3.connect('./auglab_bot.db')
            cursor = conn.cursor()
            cursor.execute("SELECT model, SUM(cost) as total_cost FROM cost_entries WHERE timestamp > datetime('now', '-1 day') GROUP BY model ORDER BY total_cost DESC")
            results = cursor.fetchall()
            conn.close()
            
            if results:
                print(f"  Recent Usage (24h):")
                for model, cost in results:
                    print(f"    {model}: ${cost:.4f}")
        except:
            pass
    
    def show_models(self):
        """Show available models."""
        print("🧠 Available Models:")
        print(f"💰 Cheap: {self.cheap_model_config.get('name', 'unknown')} ({self.cheap_model_config.get('provider', 'unknown')})")
        print(f"💎 Expensive: {self.expensive_model_config.get('name', 'unknown')} ({self.expensive_model_config.get('provider', 'unknown')})")
        
        available_models = self.config.list_available_models()
        if available_models:
            print(f"🎯 All Available: {', '.join(available_models)}")

async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Simulate @bot mentions from command line')
    parser.add_argument('message', nargs='*', help='Message to send to the bot')
    parser.add_argument('--force-expensive', action='store_true', help='Force use of expensive model')
    parser.add_argument('--force-cheap', action='store_true', help='Force use of cheap model')
    parser.add_argument('--budget', action='store_true', help='Show budget status')
    parser.add_argument('--models', action='store_true', help='Show available models')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize simulator
    print("🚀 Initializing Bot Mention Simulator...")
    simulator = BotMentionSimulator()
    
    # Handle special flags
    if args.budget:
        simulator.show_budget()
        return
    
    if args.models:
        simulator.show_models()
        return
    
    # Interactive mode
    if args.interactive:
        print("🤖 Interactive Bot Mention Simulator")
        print("Type 'quit' to exit, 'budget' for budget status, 'models' for model info")
        print("Prefix with --expensive or --cheap to force model selection")
        
        while True:
            try:
                user_input = input("\n@bot ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'budget':
                    simulator.show_budget()
                    continue
                elif user_input.lower() == 'models':
                    simulator.show_models()
                    continue
                elif not user_input:
                    continue
                
                # Parse force flags
                force_expensive = None
                if user_input.startswith('--expensive '):
                    force_expensive = True
                    user_input = user_input[12:]
                elif user_input.startswith('--cheap '):
                    force_expensive = False
                    user_input = user_input[8:]
                
                # Generate response
                print("\n" + "="*50)
                response = await simulator.generate_response(user_input, force_expensive)
                print("="*50)
                print(f"🤖 Bot Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return
    
    # Single message mode
    if args.message:
        message = ' '.join(args.message)
        force_expensive = None
        
        if args.force_expensive:
            force_expensive = True
        elif args.force_cheap:
            force_expensive = False
        
        print(f"📝 Simulating @bot mention: '{message}'")
        print("="*50)
        
        response = await simulator.generate_response(message, force_expensive)
        
        print("="*50)
        print(f"🤖 Bot Response:\n{response}")
        print("="*50)
        
    else:
        print("❌ No message provided. Use --interactive for interactive mode or provide a message.")
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main()) 