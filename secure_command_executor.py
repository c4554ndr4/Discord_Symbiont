#!/usr/bin/env python3
"""
Secure Command Executor using Docker containers for Aug Lab bot.
Provides proper sandboxing for command line access.
"""

import os
import json
import asyncio
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional
import docker
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result from a sandboxed command execution."""
    success: bool
    output: str
    error: Optional[str] = None
    return_code: int = 0
    execution_time: float = 0.0

class SecureCommandExecutor:
    """Secure command executor using Docker containers."""
    
    def __init__(self):
        """Initialize the secure executor."""
        try:
            self.docker_client = docker.from_env()
            self.enabled = True
            
            # Verify Docker is available
            self.docker_client.ping()
            logger.info("✅ Docker connection established")
            
        except Exception as e:
            logger.error(f"❌ Docker not available: {e}")
            self.enabled = False
    
    async def execute_command(self, command: str, language: str = "bash", timeout: int = 30) -> ExecutionResult:
        """Execute command in a secure Docker container."""
        if not self.enabled:
            return ExecutionResult(
                success=False,
                output="",
                error="Docker sandboxing not available"
            )
        
        try:
            # Choose appropriate Docker image
            image_map = {
                "bash": "ubuntu:22.04",
                "python": "python:3.11-slim",
                "node": "node:18-slim",
                "playwright": "mcr.microsoft.com/playwright:v1.40.0-focal"
            }
            
            image = image_map.get(language, "ubuntu:22.04")
            
            # Create isolated working directory
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # Security: Resource limits
                container_config = {
                    "image": image,
                    "command": self._build_command(command, language),
                    "working_dir": "/workspace",
                    "mem_limit": "128m",     # Memory limit
                    "cpu_period": 100000,    # CPU limit period  
                    "cpu_quota": 50000,      # 50% CPU max
                    "network_disabled": True, # No network access
                    "read_only": True,       # Read-only filesystem
                    "security_opt": ["no-new-privileges:true"],
                    "user": "nobody",        # Non-root user
                    "volumes": {
                        temp_dir: {"bind": "/workspace", "mode": "rw"}
                    },
                    "remove": True,          # Auto-cleanup
                    "stdout": True,
                    "stderr": True
                }
                
                # Execute with timeout
                start_time = asyncio.get_event_loop().time()
                
                container = self.docker_client.containers.run(
                    **container_config,
                    detach=True
                )
                
                # Wait for completion with timeout
                try:
                    result = container.wait(timeout=timeout)
                    logs = container.logs(stdout=True, stderr=True).decode('utf-8', errors='ignore')
                    
                    execution_time = asyncio.get_event_loop().time() - start_time
                    
                    return ExecutionResult(
                        success=result['StatusCode'] == 0,
                        output=logs[:2000],  # Limit output for budget
                        error=None if result['StatusCode'] == 0 else f"Exit code: {result['StatusCode']}",
                        return_code=result['StatusCode'],
                        execution_time=execution_time
                    )
                    
                except docker.errors.ContainerError as e:
                    return ExecutionResult(
                        success=False,
                        output="",
                        error=f"Container error: {e}"
                    )
                    
        except Exception as e:
            logger.error(f"Error in secure execution: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {e}"
            )
    
    def _build_command(self, command: str, language: str) -> str:
        """Build the appropriate command for the container."""
        if language == "bash":
            return ["/bin/bash", "-c", command]
        elif language == "python":
            return ["python3", "-c", command]
        elif language == "node":
            return ["node", "-e", command]
        elif language == "playwright":
            # Special handling for Playwright scripts
            return ["/bin/bash", "-c", f"echo '{command}' > script.js && node script.js"]
        else:
            return ["/bin/bash", "-c", command]
    
    async def search_web_with_playwright(self, query: str, sites: list = None) -> ExecutionResult:
        """Use Playwright to search specific sites for Aug Lab strategic info."""
        if not self.enabled:
            return ExecutionResult(success=False, output="", error="Docker not available")
        
        sites = sites or [
            "news.ycombinator.com",
            "techcrunch.com", 
            "mit.edu",
            "reddit.com/r/MachineLearning"
        ]
        
        playwright_script = f"""
const {{ chromium }} = require('playwright');

(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();
    
    const results = [];
    const query = "{query}";
    const sites = {json.dumps(sites)};
    
    for (const site of sites) {{
        try {{
            await page.goto(`https://www.google.com/search?q=site:${{site}} ${{query}}`);
            await page.waitForTimeout(2000);
            
            const searchResults = await page.$$eval('h3', elements => 
                elements.slice(0, 3).map(el => ({{
                    title: el.textContent,
                    link: el.closest('a')?.href || ''
                }}))
            );
            
            results.push({{
                site: site,
                results: searchResults
            }});
            
        }} catch (error) {{
            console.log(`Error searching ${{site}}: ${{error.message}}`);
        }}
    }}
    
    console.log(JSON.stringify(results, null, 2));
    await browser.close();
}})();
"""
        
        return await self.execute_command(playwright_script, "playwright", timeout=60)

# Global instance
secure_executor = SecureCommandExecutor()

# Integration functions for the Aug Lab bot
async def secure_run_command(command: str, language: str = "bash", timeout: int = 30) -> Dict[str, Any]:
    """Secure command execution for the bot."""
    result = await secure_executor.execute_command(command, language, timeout)
    
    return {
        "success": result.success,
        "command": command,
        "output": result.output,
        "error": result.error,
        "return_code": result.return_code,
        "execution_time": result.execution_time,
        "security": "docker_sandboxed"
    }

async def secure_web_search(query: str) -> Dict[str, Any]:
    """Secure web search using Playwright in Docker."""
    result = await secure_executor.search_web_with_playwright(query)
    
    return {
        "success": result.success,
        "query": query,
        "results": result.output,
        "error": result.error,
        "security": "docker_sandboxed"
    } 