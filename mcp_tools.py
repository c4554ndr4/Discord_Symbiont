#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Tools Integration
Provides tool calling capabilities for the Augmentation Lab bot.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    content: str
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class MCPToolManager:
    """Manages MCP tool integrations."""
    
    def __init__(self):
        self.enabled = os.getenv('MCP_ENABLE', 'True').lower() == 'true'
        self.available_tools = {}
        if self.enabled:
            self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize available MCP tools."""
        # Web Search Tool
        self.available_tools['web_search'] = {
            'name': 'web_search',
            'description': 'Search the web for current information',
            'parameters': {
                'query': {'type': 'string', 'description': 'Search query'},
                'max_results': {'type': 'integer', 'description': 'Maximum results', 'default': 5}
            }
        }
        
        # Image Analysis Tool
        self.available_tools['analyze_image'] = {
            'name': 'analyze_image',
            'description': 'Analyze an image and describe its contents',
            'parameters': {
                'image_url': {'type': 'string', 'description': 'URL of the image to analyze'}
            }
        }
        
        # Code Execution Tool
        self.available_tools['execute_code'] = {
            'name': 'execute_code',
            'description': 'Execute Python code safely',
            'parameters': {
                'code': {'type': 'string', 'description': 'Python code to execute'},
                'timeout': {'type': 'integer', 'description': 'Timeout in seconds', 'default': 30}
            }
        }
        
        # File Operation Tool
        self.available_tools['file_operations'] = {
            'name': 'file_operations',
            'description': 'Perform file operations like create, read, modify files',
            'parameters': {
                'operation': {'type': 'string', 'description': 'Operation: create, read, write, delete'},
                'filepath': {'type': 'string', 'description': 'Path to the file'},
                'content': {'type': 'string', 'description': 'File content (for write operations)', 'optional': True}
            }
        }
        
        logger.info(f"Initialized {len(self.available_tools)} MCP tools")
    
    def get_available_tools(self) -> List[Dict]:
        """Get list of available tools for AI context."""
        if not self.enabled:
            return []
        return list(self.available_tools.values())
    
    async def execute_tool(self, tool_name: str, parameters: Dict) -> ToolResult:
        """Execute a tool with given parameters."""
        if not self.enabled:
            return ToolResult(False, "", "MCP tools are disabled")
        
        if tool_name not in self.available_tools:
            return ToolResult(False, "", f"Tool '{tool_name}' not found")
        
        try:
            if tool_name == 'web_search':
                return await self._web_search(parameters)
            elif tool_name == 'analyze_image':
                return await self._analyze_image(parameters)
            elif tool_name == 'execute_code':
                return await self._execute_code(parameters)
            elif tool_name == 'file_operations':
                return await self._file_operations(parameters)
            else:
                return ToolResult(False, "", f"Tool '{tool_name}' not implemented")
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolResult(False, "", f"Tool execution error: {str(e)}")
    
    async def _web_search(self, params: Dict) -> ToolResult:
        """Perform web search using DuckDuckGo."""
        try:
            query = params.get('query', '')
            max_results = params.get('max_results', 5)
            
            # Use DuckDuckGo Instant Answer API (free)
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            
            response = requests.get(search_url, timeout=10)
            data = response.json()
            
            results = []
            
            # Get instant answer if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('AbstractText', ''),
                    'snippet': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', '')
                })
            
            # Get related topics
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0],
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', '')
                    })
            
            content = f"Search results for '{query}':\n\n"
            for i, result in enumerate(results[:max_results], 1):
                content += f"{i}. **{result['title']}**\n"
                content += f"   {result['snippet'][:200]}...\n"
                if result['url']:
                    content += f"   🔗 {result['url']}\n"
                content += "\n"
            
            return ToolResult(True, content, metadata={'query': query, 'results_count': len(results)})
            
        except Exception as e:
            return ToolResult(False, "", f"Web search error: {str(e)}")
    
    async def _analyze_image(self, params: Dict) -> ToolResult:
        """Analyze an image using OpenAI's vision model."""
        try:
            import openai
            image_url = params.get('image_url', '')
            
            # This would integrate with OpenAI's vision API
            # For now, return a placeholder
            content = f"Image analysis for: {image_url}\n"
            content += "Note: Image analysis requires OpenAI API integration. "
            content += "This is a placeholder implementation."
            
            return ToolResult(True, content, metadata={'image_url': image_url})
            
        except Exception as e:
            return ToolResult(False, "", f"Image analysis error: {str(e)}")
    
    async def _execute_code(self, params: Dict) -> ToolResult:
        """Execute Python code safely in a sandboxed environment."""
        try:
            code = params.get('code', '')
            timeout = params.get('timeout', 30)
            
            # For security, this should use a proper sandbox
            # This is a simplified implementation
            content = f"Code execution requested:\n```python\n{code}\n```\n\n"
            content += "Note: Code execution requires proper sandboxing setup. "
            content += "This is a placeholder implementation for security."
            
            return ToolResult(True, content, metadata={'code_length': len(code)})
            
        except Exception as e:
            return ToolResult(False, "", f"Code execution error: {str(e)}")
    
    async def _file_operations(self, params: Dict) -> ToolResult:
        """Perform file operations safely."""
        try:
            operation = params.get('operation', '')
            filepath = params.get('filepath', '')
            content = params.get('content', '')
            
            # Validate file path for security
            if '..' in filepath or filepath.startswith('/'):
                return ToolResult(False, "", "Invalid file path for security reasons")
            
            if operation == 'read':
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    return ToolResult(True, f"File content:\n```\n{file_content[:1000]}\n```", 
                                    metadata={'operation': 'read', 'file_size': len(file_content)})
                except FileNotFoundError:
                    return ToolResult(False, "", f"File not found: {filepath}")
            
            elif operation == 'write':
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return ToolResult(True, f"Successfully wrote to {filepath}", 
                                metadata={'operation': 'write', 'bytes_written': len(content)})
            
            elif operation == 'create':
                if os.path.exists(filepath):
                    return ToolResult(False, "", f"File already exists: {filepath}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content or "")
                return ToolResult(True, f"Successfully created {filepath}", 
                                metadata={'operation': 'create'})
            
            elif operation == 'delete':
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return ToolResult(True, f"Successfully deleted {filepath}", 
                                    metadata={'operation': 'delete'})
                else:
                    return ToolResult(False, "", f"File not found: {filepath}")
            
            else:
                return ToolResult(False, "", f"Unknown operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, "", f"File operation error: {str(e)}")

# Global instance
mcp_tools = MCPToolManager()

def parse_tool_calls(text: str) -> List[Dict]:
    """Parse tool calls from AI response text."""
    tool_calls = []
    
    # Look for tool call patterns like: [TOOL:web_search](query="AI research")
    import re
    pattern = r'\[TOOL:(\w+)\]\(([^)]+)\)'
    matches = re.findall(pattern, text)
    
    for tool_name, params_str in matches:
        try:
            # Parse parameters (simplified parser)
            params = {}
            for param in params_str.split(','):
                if '=' in param:
                    key, value = param.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    params[key] = value
            
            tool_calls.append({
                'tool': tool_name,
                'parameters': params
            })
        except Exception as e:
            logger.warning(f"Error parsing tool call: {e}")
    
    return tool_calls

async def execute_tool_calls(tool_calls: List[Dict]) -> str:
    """Execute a list of tool calls and return formatted results."""
    results = []
    
    for call in tool_calls:
        tool_name = call.get('tool')
        parameters = call.get('parameters', {})
        
        result = await mcp_tools.execute_tool(tool_name, parameters)
        
        if result.success:
            results.append(f"✅ **{tool_name}**: {result.content}")
        else:
            results.append(f"❌ **{tool_name}**: {result.error}")
    
    return "\n\n".join(results) if results else "No tool results." 