"""
Function Calling System for Augmentation Lab Bot
Handles parsing and execution of AI function calls
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)

class FunctionCallManager:
    """Manages function calling system for AI responses."""
    
    def __init__(self):
        self.available_functions = {}
        self._register_xml_patterns()
    
    def _register_xml_patterns(self):
        """Register XML patterns for function call parsing."""
        self.patterns = {
            'think': r'<think>(.*?)</think>',
            'query_memory': r'<query_memory>(.*?)</query_memory>', 
            'get_user_history': r'<get_user_history user_id="(\d+)".*?>(.*?)</get_user_history>',
            'store_observation': r'<store_observation user_id="([^"]*)" channel_id="([^"]*)" tags="([^"]*)">(.*?)</store_observation>',
            'update_constitution': r'<update_constitution reason="([^"]*)">(.*?)</update_constitution>',
            'get_constitution': r'<get_constitution.*?>(.*?)</get_constitution>',
            'message_user': r'<message_user(?:\s+users="(\[(?:[^\[\]"]|"[^"]*")*\]|[^"]*)")?(?:\s+name="([^"]*)")?(?:\s+user_id="([^"]*)")?(?:\s+destination="([^"]*)")?(?:\s+context="([^"]*)")?>(.*?)</message_user>',
            'tag_user': r'<tag_user user_id="(\d+)".*?>(.*?)</tag_user>',
            'google_search': r'<google_search>(.*?)</google_search>',
            'run_command': r'<run_command language="([^"]*)">(.*?)</run_command>',
            'generate_image': r'<generate_image(?:\s+quality="([^"]*)")?(?:\s+size="([^"]*)")?>(.*?)</generate_image>',
            'update_importance_criteria': r'<update_importance_criteria reason="([^"]*)">(.*?)</update_importance_criteria>',
            'get_budget': r'<get_budget.*?>(.*?)</get_budget>',
            'get_current_time': r'<get_current_time.*?>(.*?)</get_current_time>',
            'send_message': r'<send_message(?:\s+channel_id="([^"]*)")?(?:\s+channel_name="([^"]*)")?>(.*?)</send_message>',
            'get_available_channels': r'<get_available_channels.*?>(.*?)</get_available_channels>',
            'get_residency_members': r'<get_residency_members.*?>(.*?)</get_residency_members>',
            'find_member_by_name': r'<find_member_by_name(?:\s+name="([^"]*)")?>(.*?)</find_member_by_name>',
            'find_and_tag_user': r'<find_and_tag_user(?:\s+name="([^"]*)")?>(.*?)</find_and_tag_user>',
            'send_message_to_channel': r'<send_message_to_channel(?:\s+channel="([^"]*)")?>(.*?)</send_message_to_channel>',
            'send_direct_message': r'<send_direct_message(?:\s+user_id="([^"]*)")?>(.*?)</send_direct_message>',
            'find_and_dm_user': r'<find_and_dm_user(?:\s+name="([^"]*)")?>(.*?)</find_and_dm_user>',

            'react_to_message': r'<react_to_message(?:\s+emoji="([^"]*)")?>(.*?)</react_to_message>',
            'stop_autonomous_mode': r'<stop_autonomous_mode(?:\s+reason="([^"]*)")?>(.*?)</stop_autonomous_mode>',
            'summarize_context': r'<summarize_context(?:\s+context_type="([^"]*)")?>(.*?)</summarize_context>',
            'trigger_autonomous_mode': r'<trigger_autonomous_mode(?:\s+reason="([^"]*)")?>(.*?)</trigger_autonomous_mode>',
            'edit_memory': r'<edit_memory memory_id="([^"]*)" reason="([^"]*)">(.*?)</edit_memory>',
            'delete_memory': r'<delete_memory memory_id="([^"]*)" reason="([^"]*)">(.*?)</delete_memory>',
            'create_script': r'<create_script(?:\s+filename="([^"]*)")?(?:\s+language="([^"]*)")?>(.*?)</create_script>',
            'stop': r'<stop.*?>(.*?)</stop>'
        }
    
    def register_function(self, name: str, func: Callable):
        """Register a function that can be called by AI."""
        self.available_functions[name] = func
    
    def register_functions(self, functions: Dict[str, Callable]):
        """Register multiple functions at once."""
        self.available_functions.update(functions)
    
    def parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse XML-style command tags from AI response."""
        function_calls = []
        
        # Parse different command patterns
        for func_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                if func_name == 'think':
                    function_calls.append({
                        'function': 'think',
                        'parameters': {
                            'content': match.strip()
                        }
                    })
                elif func_name == 'query_memory':
                    function_calls.append({
                        'function': 'query_memory',
                        'parameters': {
                            'query': match.strip()
                        }
                    })
                elif func_name == 'get_user_history':
                    user_id, limit = match
                    function_calls.append({
                        'function': 'get_user_history',
                        'parameters': {
                            'user_id': int(user_id),
                            'limit': int(limit.strip()) if limit.strip().isdigit() else 20
                        }
                    })
                elif func_name == 'store_observation':
                    user_id, channel_id, tags, content = match
                    # Convert user_id to int, handle channel_id as either int or string 
                    try:
                        parsed_user_id = int(user_id) if user_id.strip().isdigit() else 0
                    except (ValueError, AttributeError):
                        parsed_user_id = 0
                    
                    try:
                        parsed_channel_id = int(channel_id) if channel_id.strip().isdigit() else 0
                    except (ValueError, AttributeError):
                        parsed_channel_id = 0
                        
                    function_calls.append({
                        'function': 'store_observation',
                        'parameters': {
                            'user_id': parsed_user_id,
                            'channel_id': parsed_channel_id,
                            'tags': tags.strip(),
                            'content': content.strip()
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
                elif func_name == 'message_user':
                    users_attr, name_attr, user_id_attr, destination_attr, context_attr, message = match
                    params = {'message': message.strip()}
                    
                    if users_attr:
                        try:
                            # First try standard JSON parsing
                            params['users'] = json.loads(users_attr)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in users attribute: {users_attr}")
                            # Try to fix common formatting issues
                            fixed_attr = users_attr
                            
                            # Fix HTML entities
                            fixed_attr = fixed_attr.replace('&quot;', '"')
                            fixed_attr = fixed_attr.replace('&amp;', '&')
                            fixed_attr = fixed_attr.replace('&lt;', '<')
                            fixed_attr = fixed_attr.replace('&gt;', '>')
                            
                            # Fix single quotes to double quotes
                            if fixed_attr.startswith('[') and fixed_attr.endswith(']'):
                                fixed_attr = fixed_attr.replace("'", '"')
                            
                            try:
                                params['users'] = json.loads(fixed_attr)
                                logger.info(f"Successfully parsed users after fixing: {fixed_attr}")
                            except json.JSONDecodeError:
                                # If it's just a plain string, wrap it in an array
                                if not users_attr.startswith('['):
                                    # Remove quotes if present and wrap in array
                                    clean_name = users_attr.strip('"\'')
                                    params['users'] = [clean_name]
                                    logger.info(f"Parsed single user as array: {clean_name}")
                                else:
                                    # Last resort: fallback to empty list
                                    logger.error(f"Could not parse users attribute even after fixes: {users_attr}")
                                    params['users'] = []
                    if name_attr:
                        params['name'] = name_attr
                    if user_id_attr:
                        try:
                            params['user_id'] = int(user_id_attr)
                        except ValueError:
                            logger.warning(f"Invalid user_id for message_user: {user_id_attr}")
                    if destination_attr:
                        params['destination'] = destination_attr
                    if context_attr:
                        params['context'] = context_attr
                    
                    function_calls.append({
                        'function': 'message_user',
                        'parameters': params
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
                            'quality': quality if quality else 'medium',
                            'size': size if size else '1024x1024'
                        }
                    })
                elif func_name == 'update_importance_criteria':
                    reason, criteria = match
                    function_calls.append({
                        'function': 'update_importance_criteria',
                        'parameters': {
                            'reason': reason,
                            'criteria': criteria.strip()
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
                elif func_name == 'get_available_channels':
                    function_calls.append({
                        'function': 'get_available_channels',
                        'parameters': {}
                    })
                elif func_name == 'get_residency_members':
                    function_calls.append({
                        'function': 'get_residency_members',
                        'parameters': {}
                    })
                elif func_name == 'find_member_by_name':
                    name_attr, content = match
                    # Use name from attribute or content
                    name = name_attr if name_attr else content.strip()
                    function_calls.append({
                        'function': 'find_member_by_name',
                        'parameters': {
                            'name': name
                        }
                    })
                elif func_name == 'find_and_tag_user':
                    name_attr, content = match
                    # Use name from attribute or content
                    name = name_attr if name_attr else content.strip()
                    function_calls.append({
                        'function': 'find_and_tag_user',
                        'parameters': {
                            'name': name,
                            'context': content.strip()
                        }
                    })
                elif func_name == 'send_message':
                    channel_id, channel_name, message = match
                    params = {'message': message.strip()}
                    
                    if channel_id:
                        # Try to convert to int if it's numeric, otherwise keep as string
                        try:
                            params['channel_id'] = int(channel_id)
                        except ValueError:
                            params['channel_id'] = channel_id
                    
                    if channel_name:
                        params['channel_name'] = channel_name
                    
                    function_calls.append({
                        'function': 'send_message',
                        'parameters': params
                    })
                elif func_name == 'send_message_to_channel':
                    channel_attr, content = match
                    channel_name = channel_attr if channel_attr else content.split(' ')[0] if content else ''
                    message = ' '.join(content.split(' ')[1:]) if content else ''
                    
                    function_calls.append({
                        'function': 'send_message_to_channel',
                        'parameters': {
                            'channel_name': channel_name.strip(),
                            'message': message.strip(),
                            'requester_name': ''  # Will be filled in by the system
                        }
                    })
                elif func_name == 'send_direct_message':
                    user_id_attr, message = match
                    user_id = user_id_attr if user_id_attr else ''
                    try:
                        user_id_int = int(user_id) if user_id else None
                    except ValueError:
                        user_id_int = None
                    function_calls.append({
                        'function': 'send_direct_message',
                        'parameters': {
                            'user_id': user_id_int,
                            'message': message.strip()
                        }
                    })
                elif func_name == 'find_and_dm_user':
                    name_attr, content = match
                    name = name_attr if name_attr else content.strip()
                    function_calls.append({
                        'function': 'find_and_dm_user',
                        'parameters': {
                            'name': name,
                            'message': content.strip()
                        }
                    })

                elif func_name == 'react_to_message':
                    emoji_attr, content = match
                    # Use emoji from attribute or content
                    emoji = emoji_attr if emoji_attr else content.strip()
                    
                    function_calls.append({
                        'function': 'react_to_message',
                        'parameters': {
                            'emoji': emoji,
                            'context': content.strip() if not emoji_attr else ''
                        }
                    })
                elif func_name == 'stop_autonomous_mode':
                    reason, summary = match
                    function_calls.append({
                        'function': 'stop_autonomous_mode',
                        'parameters': {
                            'reason': reason if reason else 'manual_stop',
                            'summary': summary.strip()
                        }
                    })
                elif func_name == 'summarize_context':
                    context_type, content = match
                    function_calls.append({
                        'function': 'summarize_context',
                        'parameters': {
                            'context_type': context_type if context_type else 'general',
                            'content': content.strip()
                        }
                    })
                elif func_name == 'trigger_autonomous_mode':
                    reason, analysis = match
                    function_calls.append({
                        'function': 'trigger_autonomous_mode',
                        'parameters': {
                            'reason': reason if reason else 'ai_decision',
                            'analysis': analysis.strip()
                        }
                    })
                elif func_name == 'edit_memory':
                    memory_id, reason, content = match
                    function_calls.append({
                        'function': 'edit_memory',
                        'parameters': {
                            'memory_id': memory_id.strip(),
                            'reason': reason.strip(),
                            'content': content.strip()
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
                elif func_name == 'create_script':
                    filename, language, content = match
                    function_calls.append({
                        'function': 'create_script',
                        'parameters': {
                            'filename': filename if filename else 'script.py',
                            'language': language if language else 'python',
                            'content': content.strip()
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
    
    def clean_xml_tags(self, text: str) -> str:
        """Remove all XML function call tags from response text."""
        original_text = text
        cleaned = text
        removed_tags = []
        
        # First, check for any unrecognized XML tags
        unrecognized_tags = self._detect_unrecognized_xml_tags(text)
        if unrecognized_tags:
            error_msg = f"❌ **Error**: Unrecognized function call(s) detected: {', '.join(unrecognized_tags)}. Available functions: {', '.join(self.available_functions.keys())}"
            logger.error(f"🚨 Unrecognized XML tags found: {unrecognized_tags}")
            return error_msg
        
        for tag_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, flags=re.DOTALL)
            if matches:
                removed_tags.append(f"{tag_name}: {len(matches)} occurrence(s)")
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        # Log what was removed if any tags were found
        if removed_tags:
            logger.info(f"🧹 Cleaned XML tags: {', '.join(removed_tags)}")
            logger.info(f"📏 Text length: {len(original_text)} → {len(cleaned)} chars")
        
        return cleaned
    
    def _detect_unrecognized_xml_tags(self, text: str) -> List[str]:
        """Detect any XML-like tags that aren't in our recognized patterns or are malformed."""
        import re
        
        unrecognized = []
        recognized_tag_names = set(self.patterns.keys())
        
        # First check for malformed XML (unquoted attributes)
        malformed_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)[^>]*=[^"\s][^>\s]*[^>]*>'
        malformed_matches = re.findall(malformed_pattern, text)
        for tag_name in malformed_matches:
            if tag_name in recognized_tag_names:
                unrecognized.append(f"{tag_name} (MALFORMED: attributes must be quoted)")
        
        # Then check for unrecognized tags with proper XML format
        all_xml_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)[^>]*>(.*?)</\1>'
        all_tags = re.findall(all_xml_pattern, text, re.DOTALL)
        
        for tag_name, _ in all_tags:
            if tag_name not in recognized_tag_names:
                if f"{tag_name} (MALFORMED: attributes must be quoted)" not in unrecognized:
                    unrecognized.append(tag_name)
        
        return unrecognized
    
    def get_function_list(self) -> List[str]:
        """Get list of available function names."""
        return list(self.available_functions.keys()) 