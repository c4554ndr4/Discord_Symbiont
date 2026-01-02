#!/usr/bin/env python3
"""
Simple health check server for Render deployment  
Runs alongside the Discord bot without threading conflicts
"""

import asyncio
import aiohttp
from aiohttp import web
import logging
import os

logger = logging.getLogger(__name__)

class HealthCheckServer:
    def __init__(self, port=8080):
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.runner = None
        
    def setup_routes(self):
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/', self.root)
    
    async def health_check(self, request):
        """Health check endpoint for Render"""
        return web.json_response({
            'status': 'healthy',
            'service': 'auglab-discord-bot',
            'message': 'Bot is running with OpenAI embeddings'
        })
    
    async def root(self, request):
        """Root endpoint"""
        return web.json_response({
            'service': 'Augmentation Lab Discord Bot',
            'status': 'running',
            'embeddings': 'OpenAI API'
        })
    
    async def start_server(self):
        """Start the health check server using asyncio"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await site.start()
            logger.info(f"✅ Health check server started on port {self.port}")
        except Exception as e:
            logger.error(f"❌ Health check server error: {e}")
    
    async def cleanup(self):
        """Cleanup the server"""
        if self.runner:
            await self.runner.cleanup()

# Global server instance
health_server = None

async def start_health_server():
    """Start health check server using async"""
    global health_server
    try:
        health_server = HealthCheckServer()
        await health_server.start_server()
        return True
    except Exception as e:
        logger.error(f"Failed to start health server: {e}")
        return False

if __name__ == '__main__':
    async def main():
        await start_health_server()
        # Keep running
        while True:
            await asyncio.sleep(60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Health check server stopped") 