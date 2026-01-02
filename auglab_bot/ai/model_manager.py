"""
Model Management for Augmentation Lab Bot
Handles model configuration, API clients, and model switching
"""

import os
import logging
import tiktoken
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

import anthropic
import openai

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages AI model configurations and API clients."""
    
    def __init__(self, config_manager):
        self.config = config_manager
        
        # Model configurations
        self.cheap_model_config = self.config.get_model_config('cheap')
        self.expensive_model_config = self.config.get_model_config('expensive')
        
        # Initialize API clients
        self.anthropic_client = None
        self.openai_client = None
        self.openrouter_client = None
        self.groq_client = None
        
        # Initialize tokenizer for cost estimation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize clients
        self._init_api_clients()
    
    def _init_api_clients(self):
        """Initialize API clients based on configuration."""
        try:
            # Anthropic client
            anthropic_key = self.config.get('api_keys.anthropic')
            if anthropic_key:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            
            # OpenAI client
            openai_key = self.config.get('api_keys.openai')
            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)
            
            # OpenRouter client
            openrouter_key = self.config.get('api_keys.openrouter')
            if openrouter_key:
                self.openrouter_client = openai.OpenAI(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            
            # Groq client
            groq_key = self.config.get('api_keys.groq')
            if groq_key:
                self.groq_client = openai.OpenAI(
                    api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {e}")
            raise
    
    async def call_model_api(self, model_config: Dict, prompt: str, max_tokens: int = 800) -> str:
        """Call the appropriate API based on model configuration."""
        provider = model_config.get('provider', 'anthropic')
        model = model_config.get('model', 'claude-3-haiku-20240307')
        
        if provider == 'anthropic':
            if not self.anthropic_client:
                raise Exception("Anthropic client not initialized")
            
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
            
        elif provider == 'groq':
            if not self.groq_client:
                raise Exception("Groq client not initialized - check GROQ_API_KEY")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.groq_client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.choices[0].message.content
            
        elif provider == 'openai':
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")
            
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

    async def call_model_api_with_conversation(self, model_config: Dict, system_message: str, 
                                             conversation_messages: List[Dict[str, str]], 
                                             max_tokens: int = 800) -> str:
        """Call the appropriate API with proper conversation structure."""
        provider = model_config.get('provider', 'anthropic')
        model = model_config.get('model', 'claude-3-haiku-20240307')
        
        # Get token limits and stop sequences for the provider
        token_limits = self._get_token_limits(provider)
        effective_max_tokens = token_limits.get('max_tokens', max_tokens)
        stop_sequences = token_limits.get('stop_sequences', [])
        
        logger.info(f"Using {provider} with max_tokens: {effective_max_tokens}, stop_sequences: {stop_sequences}")
        
        if provider == 'anthropic':
            if not self.anthropic_client:
                raise Exception("Anthropic client not initialized")
            
            # Anthropic uses system parameter + messages
            kwargs = {
                'model': model,
                'max_tokens': effective_max_tokens,
                'system': system_message,
                'messages': conversation_messages
            }
            if stop_sequences:
                kwargs['stop_sequences'] = stop_sequences
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(**kwargs)
            )
            return response.content[0].text
            
        elif provider == 'openrouter':
            if not self.openrouter_client:
                raise Exception("OpenRouter client not initialized - check OPEN_ROUTER_KEY")
            
            # OpenRouter uses system message as first message
            messages = [{"role": "system", "content": system_message}] + conversation_messages
            kwargs = {
                'model': model,
                'max_tokens': effective_max_tokens,
                'messages': messages
            }
            if stop_sequences:
                kwargs['stop'] = stop_sequences
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openrouter_client.chat.completions.create(**kwargs)
            )
            return response.choices[0].message.content
            
        elif provider == 'groq':
            if not self.groq_client:
                raise Exception("Groq client not initialized - check GROQ_API_KEY")
            
            # Groq uses system message as first message (OpenAI-compatible)
            messages = [{"role": "system", "content": system_message}] + conversation_messages
            kwargs = {
                'model': model,
                'max_tokens': effective_max_tokens,
                'messages': messages
            }
            if stop_sequences:
                kwargs['stop'] = stop_sequences
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.groq_client.chat.completions.create(**kwargs)
            )
            return response.choices[0].message.content
            
        elif provider == 'openai':
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")
            
            # OpenAI uses system message as first message
            messages = [{"role": "system", "content": system_message}] + conversation_messages
            kwargs = {
                'model': model,
                'max_tokens': effective_max_tokens,
                'messages': messages
            }
            if stop_sequences:
                kwargs['stop'] = stop_sequences
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(**kwargs)
            )
            return response.choices[0].message.content
            
        else:
            raise Exception(f"Unsupported provider: {provider}")
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        return len(self.tokenizer.encode(text))
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model configuration."""
        if model_type == 'cheap':
            return self.cheap_model_config
        elif model_type == 'expensive':
            return self.expensive_model_config
        else:
            return {}
    
    def _get_token_limits(self, provider: str) -> Dict[str, Any]:
        """Get token limits configuration for a provider."""
        token_limits_config = self.config.get('models.token_limits', {})
        provider_config = token_limits_config.get(provider, {})
        
        # Return defaults if provider not configured or disabled
        if not provider_config.get('enabled', False):
            return {'max_tokens': 1000, 'stop_sequences': []}
        
        return {
            'max_tokens': provider_config.get('max_tokens', 1000),
            'stop_sequences': provider_config.get('stop_sequences', [])
        }
    
    def switch_model(self, model_type: str, model_name: str) -> bool:
        """Switch cheap or expensive model to a different configuration."""
        return self.config.switch_model(model_type, model_name)
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        return self.config.list_available_models() 