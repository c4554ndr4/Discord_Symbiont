"""
Response Generation for Augmentation Lab Bot
Handles AI response generation with importance assessment and model selection
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..storage.cost_tracker import CostEntry

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles AI response generation with intelligent model selection."""
    
    def __init__(self, model_manager, function_manager, conversation_manager, 
                 cost_tracker, memory_system, working_memory_manager, config_manager):
        self.model_manager = model_manager
        self.function_manager = function_manager
        self.conversation_manager = conversation_manager
        self.cost_tracker = cost_tracker
        self.memory_system = memory_system
        self.working_memory_manager = working_memory_manager
        self.config = config_manager
        
        # Bot reference (set after initialization)
        self.bot = None
        self.function_implementations = None
        
        # Importance criteria (can be updated by AI)
        self.importance_criteria = self._get_default_importance_criteria()
    
    def _get_default_importance_criteria(self) -> str:
        """Get default importance criteria for message assessment."""
        return """
Use expensive Sonnet model when:
- Strategic planning or Aug Lab metrics (social media, projects, satisfaction)
- Complex technical questions requiring detailed analysis
- Multi-step problem solving or debugging
- Creative tasks requiring high-quality output
- Questions about the bot's constitution or advanced features
- User seems frustrated or needs careful handling

Use cheap Haiku model when:
- Simple questions with clear answers
- Basic information requests
- Casual conversation
- Repetitive or low-stakes interactions
- Budget is running low
"""
    
    def get_importance_criteria(self) -> str:
        """Get current importance criteria."""
        return self.importance_criteria
    
    def update_importance_criteria(self, new_criteria: str, reason: str = "") -> Dict[str, Any]:
        """Update importance criteria for message assessment."""
        self.importance_criteria = new_criteria
        logger.info(f"Updated importance criteria: {reason}")
        return {
            "success": True,
            "message": "Importance criteria updated successfully",
            "reason": reason
        }
    
    async def assess_importance(self, message, content: str) -> Dict[str, Any]:
        """Use Haiku to assess if a message needs expensive model treatment."""
        try:
            assessment_prompt = f"""You are a message importance filter for the Aug Lab Discord bot. Your job is to decide if this message needs the expensive Sonnet model or if Haiku can handle it.

## CURRENT IMPORTANCE CRITERIA
{self.importance_criteria}

## MESSAGE TO ASSESS
Message: "{content}"
Author: {message.author.display_name if hasattr(message, 'author') else 'Unknown'}
Channel: #{message.channel.name if hasattr(message, 'channel') and hasattr(message.channel, 'name') else 'DM'}
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

            input_tokens = self.model_manager.get_token_count(assessment_prompt)
            
            if not self.cost_tracker.can_afford('haiku', input_tokens + 100):
                return {'use_expensive_model': False, 'reasoning': 'Budget constraint'}
            
            cheap_config = self.model_manager.get_model_config('cheap')
            response = await self.model_manager.call_model_api(
                cheap_config, assessment_prompt, max_tokens=200
            )
            
            output_tokens = self.model_manager.get_token_count(response)
            cost = self.cost_tracker.calculate_cost('haiku', input_tokens, output_tokens)
            
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model='haiku',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                action='importance_assessment',
                user_id=message.author.id if hasattr(message, 'author') else 0,
                channel_id=message.channel.id if hasattr(message, 'channel') else 0
            ))
            
            # Try to parse JSON response
            try:
                assessment = json.loads(response.strip())
                return assessment
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    'use_expensive_model': False,
                    'reasoning': 'JSON parsing failed - using cheap model as fallback'
                }
                
        except Exception as e:
            logger.error(f"Error in importance assessment: {e}")
            return {'use_expensive_model': False, 'reasoning': f'Assessment error: {str(e)}'}
    
    async def generate_response(self, message, content: str, use_expensive: bool = False) -> Optional[str]:
        """Generate AI response using appropriate model with conversation threading."""
        try:
            model_config = self.model_manager.get_model_config('expensive' if use_expensive else 'cheap')
            model_name = model_config.get('name', 'haiku')
            
            # Log recent context info
            recent_context = self.conversation_manager.get_recent_context(message)
            logger.info(f"💬 Recent context for {getattr(message.author, 'display_name', 'Unknown')}: {len(recent_context)} messages")
            
            # Add user message to recent context
            self.conversation_manager.add_message_to_context(message, 'user', content)
            
            # Build system message with context
            logger.info(f"🧠 Building system message with memory integration...")
            system_message = self.conversation_manager.build_system_message(
                message, self.config, self.working_memory_manager, self.memory_system
            )
            logger.info(f"🧠 System message complete: {len(system_message)} characters")
            
            # Build conversation messages
            conversation_messages = self.conversation_manager.build_conversation_messages(
                message, content, use_expensive
            )
            
            # Estimate tokens for both system and conversation messages
            system_tokens = self.model_manager.get_token_count(system_message)
            conversation_tokens = sum(
                self.model_manager.get_token_count(msg['content']) 
                for msg in conversation_messages
            )
            total_input_tokens = system_tokens + conversation_tokens
            
            max_tokens = 1500 if use_expensive else 800
            
            if not self.cost_tracker.can_afford(model_name, total_input_tokens + max_tokens):
                return "⚠️ Insufficient budget for this request. Use `!budget` to check remaining funds."
            
            # Make API call with proper conversation structure
            response_text = await self.model_manager.call_model_api_with_conversation(
                model_config, system_message, conversation_messages, max_tokens
            )
            
            # Add assistant response to recent context
            self.conversation_manager.add_message_to_context(message, 'assistant', response_text)
            
            output_tokens = self.model_manager.get_token_count(response_text)
            cost = self.cost_tracker.calculate_cost(model_name, total_input_tokens, output_tokens)
            
            # Record cost
            self.cost_tracker.record_usage(CostEntry(
                timestamp=datetime.now(),
                model=model_name,
                input_tokens=total_input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                action='chat_response',
                user_id=message.author.id if hasattr(message, 'author') else 0,
                channel_id=message.channel.id if hasattr(message, 'channel') else 0
            ))
            
            # Store interaction in memory
            interaction_content = f"User: {content}\nBot: {response_text}"
            self.memory_system.store_memory(
                interaction_content,
                message.author.id if hasattr(message, 'author') else 0,
                message.channel.id if hasattr(message, 'channel') else 0,
                memory_type='interaction',
                tags=['chat', 'conversation']
            )
            
            # Check for and execute function calls
            function_calls = self.function_manager.parse_function_calls(response_text)
            if function_calls:
                logger.info(f"🔧 Executing {len(function_calls)} function call(s): {[call['function'] for call in function_calls]}")
                
                # Log detailed function call information
                for i, call in enumerate(function_calls, 1):
                    logger.info(f"📋 Function Call {i}: {call['function']}")
                    if call.get('parameters'):
                        logger.info(f"   Parameters: {call['parameters']}")
                    if call.get('content'):
                        logger.info(f"   Content: {call['content']}")
                
                # Execute function calls and get results
                function_results = await self.function_manager.execute_function_calls(function_calls)
                logger.info(f"✅ Function calls completed, generating response with results...")
                
                # Log function results
                logger.info("📤 FUNCTION RESULTS:")
                logger.info(function_results)
                
                # Extract and store user tags for channel messaging
                user_tags = set()
                for call in function_calls:
                    if call.get('function') == 'tag_user':
                        user_id = call.get('parameters', {}).get('user_id')
                        if user_id:
                            user_tags.add(f"<@{user_id}>")
                
                # Also extract from message_user and find_and_tag_user results
                import re
                
                # Extract single user_tag (backwards compatibility)
                tag_pattern = r'"user_tag":\s*"(<@\d+>)"'
                tag_matches = re.findall(tag_pattern, function_results)
                for tag in tag_matches:
                    user_tags.add(tag)
                
                # Extract user_tags array (new format)
                tags_array_pattern = r'"user_tags":\s*\[(.*?)\]'
                tags_array_matches = re.findall(tags_array_pattern, function_results, re.DOTALL)
                for tags_array_str in tags_array_matches:
                    # Extract individual tags from the array string
                    individual_tags = re.findall(r'"(<@\d+>)"', tags_array_str)
                    for tag in individual_tags:
                        user_tags.add(tag)
                
                # Store user tags for channel messaging
                if self.function_implementations and user_tags:
                    self.function_implementations._current_user_tags = list(user_tags)
                
                # Check for pending autonomous messages and send them
                await self._handle_pending_autonomous_messages()
                
                # Check for pending user messages and send them
                await self._handle_pending_user_messages()
                
                # Create a new prompt that includes the function results
                function_context = f"""
## Function Call Results

The AI assistant just executed these function calls:
{function_results}

Please provide a natural response to the user that incorporates these function results. Be conversational and helpful, presenting the information in a user-friendly way. Don't mention the technical function call process - just naturally integrate the results into your response.

Original user message: {content}
Original AI reasoning: {response_text}
"""
                
                # Generate a new response incorporating the function results
                new_input_tokens = self.model_manager.get_token_count(function_context)
                new_max_tokens = 800 if use_expensive else 500
                
                if self.cost_tracker.can_afford(model_name, new_input_tokens + new_max_tokens):
                    # Response generation with length validation and retry
                    final_response = await self._generate_response_with_validation(
                        model_config, function_context, new_max_tokens, model_name, 
                        new_input_tokens, message, content
                    )
                    
                    # Process any user tags from function results (deduplicate)
                    user_tags = set()  # Use set to prevent duplicates
                    tag_calls = 0
                    
                    # Process tag_user calls (from parameters)
                    for call in function_calls:
                        if call.get('function') == 'tag_user':
                            tag_calls += 1
                            user_id = call.get('parameters', {}).get('user_id')
                            if user_id:
                                user_tags.add(f"<@{user_id}>")
                    
                    # Process message_user and find_and_tag_user calls (from results)
                    # Parse function results to extract user tags
                    import re
                    
                    # Extract single user_tag (backwards compatibility)
                    tag_pattern = r'"user_tag":\s*"(<@\d+>)"'
                    tag_matches = re.findall(tag_pattern, function_results)
                    for tag in tag_matches:
                        user_tags.add(tag)
                        tag_calls += 1
                    
                    # Extract user_tags array (new format)
                    tags_array_pattern = r'"user_tags":\s*\[(.*?)\]'
                    tags_array_matches = re.findall(tags_array_pattern, function_results, re.DOTALL)
                    for tags_array_str in tags_array_matches:
                        # Extract individual tags from the array string
                        individual_tags = re.findall(r'"(<@\d+>)"', tags_array_str)
                        for tag in individual_tags:
                            user_tags.add(tag)
                            tag_calls += 1
                    
                    # Log tagging behavior for debugging
                    if tag_calls > 0:
                        logger.info(f"🏷️ Processed {tag_calls} tag_user calls → {len(user_tags)} unique user tags")
                    
                    # Add user tags to response if any (convert back to list for joining)
                    if user_tags:
                        final_response = f"{' '.join(list(user_tags))} {final_response}".strip()
                    
                    # Validate final response length (including tags)
                    final_response = self._validate_response_length(final_response)
                    
                    # Return cleaned response
                    cleaned_response = self.function_manager.clean_xml_tags(final_response)
                    logger.info("🎯 FINAL RESPONSE TO USER (after function calls & tagging):")
                    logger.info(cleaned_response)
                    return cleaned_response
                else:
                    logger.warning("Can't afford function result integration - returning original response")
                    cleaned_response = self.function_manager.clean_xml_tags(response_text)
                    # Validate response length even without function integration
                    cleaned_response = self._validate_response_length(cleaned_response)
                    logger.info("🎯 FINAL RESPONSE TO USER (budget limited):")
                    logger.info(cleaned_response)
                    return cleaned_response
            
            # Return cleaned response (remove function call tags)
            cleaned_response = self.function_manager.clean_xml_tags(response_text)
            # Validate response length for non-function responses
            cleaned_response = self._validate_response_length(cleaned_response)
            logger.info("🎯 FINAL RESPONSE TO USER (no function calls):")
            logger.info(cleaned_response)
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    async def handle_ai_response(self, message, force_expensive: Optional[bool] = None):
        """Handle AI responses with intelligent model selection."""
        logger.info(f"🤖 Starting AI response for {getattr(message.author, 'display_name', 'Unknown')}: '{getattr(message, 'content', '')[:100]}...'")
        
        # Set current message for reactions and other functions
        if self.function_implementations:
            self.function_implementations._current_message = message
        
        if self.cost_tracker.get_remaining_budget() <= 0:
            logger.warning("❌ Budget exhausted!")
            return "⚠️ Budget exhausted! Cannot process AI requests until reset."
            
        # Clean message content
        content = getattr(message, 'content', '')
        
        # Remove bot mentions
        if hasattr(message, 'mentions'):
            for mention in message.mentions:
                content = content.replace(f'<@{mention.id}>', '').strip()
        
        # Remove prefix if present
        prefix = self.config.get('bot.prefix', '!')
        if content.startswith(prefix):
            content = content[len(prefix):].strip()
            
        if not content:
            content = "Hello!"
            
        logger.info(f"🧹 Cleaned content: '{content}'")
        
        # Determine model selection
        if force_expensive is not None:
            # Strategic monitor has already decided
            use_expensive = force_expensive and self.cost_tracker.can_afford('sonnet', 2000)
            logger.info(f"📊 Model selection forced: {'Sonnet' if use_expensive else 'Haiku'}")
        else:
            # Use importance assessment
            logger.info("📊 Assessing message importance...")
            importance_assessment = await self.assess_importance(message, content)
            use_expensive = importance_assessment['use_expensive_model'] and self.cost_tracker.can_afford('sonnet', 2000)
            logger.info(f"📊 Importance assessment: {importance_assessment}")
            logger.info(f"📊 Using model: {'Sonnet' if use_expensive else 'Haiku'}")
        
        logger.info("🎯 Generating response...")
        response = await self.generate_response(message, content, use_expensive=use_expensive)
        logger.info(f"📝 Generated response ({len(response) if response else 0} chars)")
        
        # Log the full raw response with XML tags for debugging
        if response:
            logger.info("=" * 80)
            logger.info("🔍 FULL RAW AI RESPONSE:")
            logger.info(response)
            logger.info("=" * 80)
        
        return response
    
    async def _generate_response_with_validation(self, model_config, prompt, max_tokens, model_name, input_tokens, message, original_content):
        """Generate response with length validation and retry logic for short response mode."""
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

CRITICAL: Your previous response was too long. You MUST provide a response that is {short_response_limit} characters or less (including any user tags). Be extremely concise while maintaining the key information."""
                
                # Generate response
                response = await self.model_manager.call_model_api(
                    model_config, current_prompt, max_tokens
                )
                
                # Calculate and record cost
                output_tokens = self.model_manager.get_token_count(response)
                cost = self.cost_tracker.calculate_cost(model_name, input_tokens, output_tokens)
                
                self.cost_tracker.record_usage(CostEntry(
                    timestamp=datetime.now(),
                    model=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    action='function_response',
                    user_id=message.author.id if hasattr(message, 'author') else 0,
                    channel_id=message.channel.id if hasattr(message, 'channel') else 0
                ))
                
                # Check length if short mode is enabled
                if short_response_mode and len(response) > short_response_limit:
                    if attempt < max_retries:
                        logger.warning(f"⚠️ Response too long ({len(response)} chars > {short_response_limit} limit), retrying...")
                        continue
                    else:
                        # Final attempt - truncate with warning
                        response = response[:short_response_limit-3] + "..."
                        logger.error(f"❌ Response still too long after {max_retries} retries, truncating to {short_response_limit} chars")
                
                # Log successful response
                if short_response_mode:
                    logger.info(f"✅ Response length OK: {len(response)}/{short_response_limit} characters")
                
                return response
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"⚠️ Response generation failed (attempt {attempt + 1}), retrying: {e}")
                    continue
                else:
                    logger.error(f"❌ Response generation failed after {max_retries} retries: {e}")
                    raise
    
    def _validate_response_length(self, response: str) -> str:
        """Validate and potentially truncate response based on short response mode."""
        short_response_mode = self.config.get('bot.short_response_mode', False)
        
        if not short_response_mode:
            return response
        
        short_response_limit = self.config.get('bot.short_response_limit', 240)
        
        if len(response) <= short_response_limit:
            logger.info(f"✅ Response length OK: {len(response)}/{short_response_limit} characters")
            return response
        
        # Truncate with ellipsis if too long
        truncated = response[:short_response_limit-3] + "..."
        logger.warning(f"⚠️ Response truncated: {len(response)} chars → {len(truncated)} chars (limit: {short_response_limit})")
        return truncated
    
    async def _handle_pending_autonomous_messages(self):
        """Handle any pending autonomous messages after function calls."""
        if not self.function_implementations:
            return
            
        try:
            # Check if there's a pending autonomous message
            if hasattr(self.function_implementations, '_pending_autonomous_message'):
                pending = self.function_implementations._pending_autonomous_message
                if pending and self.bot:
                    channel = pending['channel']
                    message = pending['message']
                    
                    logger.info(f"📤 Sending autonomous message to #{getattr(channel, 'name', 'DM')}")
                    await channel.send(message)
                    
                    # Store the sent message in memory
                    if self.memory_system:
                        self.memory_system.store_memory(
                            f"Sent autonomous message to #{getattr(channel, 'name', 'DM')}: {message[:100]}...",
                            user_id=0,
                            channel_id=channel.id,
                            memory_type='observation',
                            tags=['autonomous', 'message_sent']
                        )
                    
                    # Clear the pending message
                    self.function_implementations._pending_autonomous_message = None
                    logger.info("✅ Autonomous message sent successfully")
                    
        except Exception as e:
            logger.error(f"Error sending autonomous message: {e}")

    async def _handle_pending_user_messages(self):
        """Handle any pending user messages after function calls."""
        if not self.function_implementations:
            return
            
        try:
            # Check if there's a pending user message
            if hasattr(self.function_implementations, '_pending_user_message'):
                pending = self.function_implementations._pending_user_message
                if pending and self.bot:
                    channel = pending['channel']
                    message = pending['message']
                    requester = pending.get('requester', 'Unknown user')
                    
                    # Check if there are any user tags from function results to include
                    if hasattr(self.function_implementations, '_current_user_tags'):
                        user_tags = self.function_implementations._current_user_tags
                        if user_tags:
                            message = f"{' '.join(user_tags)} {message}".strip()
                    
                    logger.info(f"📤 Sending user-requested message to #{getattr(channel, 'name', 'DM')} on behalf of {requester}")
                    await channel.send(message)
                    
                    # Store the sent message in memory
                    if self.memory_system:
                        self.memory_system.store_memory(
                            f"Sent message to #{getattr(channel, 'name', 'DM')} on behalf of {requester}: {message[:100]}...",
                            user_id=0,
                            channel_id=channel.id,
                            memory_type='observation',
                            tags=['user_request', 'message_sent', 'cross_channel']
                        )
                    
                    # Clear the pending message and user tags
                    self.function_implementations._pending_user_message = None
                    if hasattr(self.function_implementations, '_current_user_tags'):
                        self.function_implementations._current_user_tags = None
                    logger.info("✅ User-requested message sent successfully")
            
            # Check if there's a pending DM
            if hasattr(self.function_implementations, '_pending_dm'):
                pending = self.function_implementations._pending_dm
                if pending and self.bot:
                    user = pending['user']
                    message = pending['message']
                    context = pending.get('context', '')
                    
                    logger.info(f"📤 Sending DM to {user.name} ({user.id})")
                    await user.send(message)
                    
                    # Store the sent DM in memory
                    if self.memory_system:
                        self.memory_system.store_memory(
                            f"Sent DM to {user.name}: {message[:100]}...",
                            user_id=user.id,
                            channel_id=0,  # DMs don't have channel IDs
                            memory_type='observation',
                            tags=['dm_sent', 'direct_message', 'user_request']
                        )
                    
                    # Clear the pending DM
                    self.function_implementations._pending_dm = None
                    logger.info(f"✅ DM sent successfully to {user.name}")
                    


            # Check if there's a pending reaction
            if hasattr(self.function_implementations, '_pending_reaction'):
                pending = self.function_implementations._pending_reaction
                if pending and self.bot:
                    message = pending['message']
                    emoji = pending['emoji']
                    context = pending.get('context', '')
                    
                    logger.info(f"📤 Adding reaction {emoji} to message")
                    try:
                        await message.add_reaction(emoji)
                        
                        # Store the reaction in memory
                        if self.memory_system:
                            self.memory_system.store_memory(
                                f"Reacted to message with {emoji}" + (f": {context}" if context else ""),
                                user_id=0,
                                channel_id=message.channel.id,
                                memory_type='observation',
                                tags=['reaction', 'emoji', 'engagement']
                            )
                        
                        # Clear the pending reaction
                        self.function_implementations._pending_reaction = None
                        logger.info(f"✅ Reaction {emoji} added successfully")
                        
                    except Exception as reaction_error:
                        logger.error(f"Failed to add reaction {emoji}: {reaction_error}")
                        self.function_implementations._pending_reaction = None
                    
        except Exception as e:
            logger.error(f"Error handling pending messages: {e}")

    async def generate_autonomous_response(self, prompt: str) -> str:
        """Generate response for autonomous mode using expensive model."""
        try:
            expensive_config = self.model_manager.get_model_config('expensive')
            input_tokens = self.model_manager.get_token_count(prompt)
            
            if not self.cost_tracker.can_afford('sonnet', input_tokens + 1000):
                logger.warning("Cannot afford autonomous thinking - skipping")
                return ""
                
            response = await self.model_manager.call_model_api(
                expensive_config, prompt, max_tokens=1500
            )
            
            output_tokens = self.model_manager.get_token_count(response)
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
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating autonomous response: {e}")
            return "" 