"""
Configuration Management for Augmentation Lab Bot
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages bot configuration with support for multiple providers (Anthropic, OpenRouter, Groq, OpenAI)."""
    
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
                if isinstance(env_var, str) and env_var.startswith('${') and env_var.endswith('}'):
                    # Handle ${ENV_VAR} format
                    env_name = env_var[2:-1]
                    env_value = os.getenv(env_name)
                    if env_value:
                        config["api_keys"][key] = env_value
                    else:
                        logger.warning(f"Environment variable {env_name} not found for {key}")
                elif isinstance(env_var, str):
                    # Handle direct environment variable names
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
            "bot_name": "Dax",
            "prefix": "!",
            "debug": True,
            "api_keys": {
                "anthropic": "${ANTHROPIC_API_KEY}",
                "openai": "${OPENAI_API_KEY}",
                "discord": "${DISCORD_TOKEN}",
                "openrouter": "${OPEN_ROUTER_KEY}",
                "groq": "${GROQ_API_KEY}"
            },
            "models": {
                "primary_provider": "openrouter",
                "cheap_model": {"provider": "groq", "model": "llama-3.1-8b-instant", "name": "llama3-8b-groq"},
                "expensive_model": {"provider": "openrouter", "model": "moonshotai/kimi-k2:free", "name": "kimi"},
                "available_models": {
                    "claude-haiku": {"provider": "anthropic", "model": "claude-3-haiku-20240307", "name": "haiku"},
                    "kimi": {"provider": "openrouter", "model": "moonshotai/kimi-k2:free", "name": "kimi"},
                    "llama3-8b-groq": {"provider": "groq", "model": "llama-3.1-8b-instant", "name": "llama3-8b-groq"}
                }
            },
            "features": {"conversation_memory": True, "dm_support": True}
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
    
    def list_models_by_provider(self, provider: str) -> List[str]:
        """List all available models for a specific provider."""
        available = self.get("models.available_models", {})
        return [name for name, config in available.items() 
                if config.get("provider") == provider]
    
    def get_available_providers(self) -> List[str]:
        """Get list of all configured providers."""
        available = self.get("models.available_models", {})
        providers = set()
        for config in available.values():
            if "provider" in config:
                providers.add(config["provider"])
        return sorted(list(providers))
    
    def switch_model(self, model_type: str, model_name: str) -> bool:
        """Switch the cheap or expensive model to a different model."""
        if model_type not in ["cheap", "expensive"]:
            logger.error(f"Invalid model type: {model_type}. Must be 'cheap' or 'expensive'")
            return False
            
        available_models = self.get("models.available_models", {})
        if model_name not in available_models:
            logger.error(f"Model {model_name} not found in available models")
            return False
            
        model_config = available_models[model_name]
        self.set(f"models.{model_type}_model", model_config)
        logger.info(f"Switched {model_type} model to {model_name} ({model_config.get('provider')})")
        return True
    
    def switch_primary_provider(self, provider: str) -> bool:
        """Switch the primary provider for the bot."""
        available_providers = self.get_available_providers()
        if provider not in available_providers:
            logger.error(f"Provider {provider} not available. Available: {available_providers}")
            return False
        
        self.set("models.primary_provider", provider)
        logger.info(f"Switched primary provider to {provider}")
        return True
    
    def get_provider_models(self, provider: str) -> Dict[str, Dict]:
        """Get all models for a specific provider."""
        available = self.get("models.available_models", {})
        return {name: config for name, config in available.items() 
                if config.get("provider") == provider}
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a model."""
        model_config = self.get_model_config(model_name)
        if not model_config:
            return None
        
        return {
            "name": model_config.get("name", model_name),
            "provider": model_config.get("provider", "unknown"),
            "model": model_config.get("model", "unknown"),
            "endpoint": model_config.get("endpoint", "default")
        }
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present."""
        api_keys = self.get("api_keys", {})
        validation = {}
        
        for provider, key in api_keys.items():
            if isinstance(key, str) and key and not key.startswith('${'):
                validation[provider] = True
            else:
                validation[provider] = False
                
        return validation
    
    def get_config_summary(self) -> Dict:
        """Get a summary of current configuration."""
        return {
            "primary_provider": self.get("models.primary_provider"),
            "cheap_model": self.get_model_info("cheap"),
            "expensive_model": self.get_model_info("expensive"),
            "available_providers": self.get_available_providers(),
            "total_models": len(self.list_available_models()),
            "api_keys_status": self.validate_api_keys()
        } 