#!/usr/bin/env python3
"""
Check member permissions and intent issues
"""

import os
import sys
import asyncio
import discord
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

from auglab_bot.config.manager import ConfigManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_member_access():
    """Check what member access the bot currently has."""
    
    config = ConfigManager()
    token = os.getenv('DISCORD_BOT_TOKEN')
    target_guild_id = config.get('bot.target_guild_id')
    
    # Try with just basic intents first
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    
    client = discord.Client(intents=intents)
    
    @client.event
    async def on_ready():
        print(f"🤖 Logged in as {client.user.name}")
        
        guild = client.get_guild(int(target_guild_id))
        if not guild:
            print("❌ Could not find target guild!")
            await client.close()
            return
            
        print(f"✅ Found guild: {guild.name}")
        print(f"👥 Reported member count: {guild.member_count}")
        print(f"📋 Cached members: {len(guild.members)}")
        
        # Check bot permissions
        bot_member = guild.get_member(client.user.id)
        if bot_member:
            perms = bot_member.guild_permissions
            print(f"\n🤖 Bot Permissions:")
            print(f"   - View Guild: {perms.view_guild}")
            print(f"   - Read Messages: {perms.read_messages}")
            print(f"   - Send Messages: {perms.send_messages}")
            print(f"   - Read Message History: {perms.read_message_history}")
            print(f"   - Use Slash Commands: {perms.use_slash_commands}")
            print(f"   - Manage Messages: {perms.manage_messages}")
            print(f"   - Embed Links: {perms.embed_links}")
            print(f"   - Attach Files: {perms.attach_files}")
            print(f"   - Add Reactions: {perms.add_reactions}")
        
        # Check what members we can see
        print(f"\n👥 Visible Members (showing first 20):")
        visible_members = list(guild.members)[:20]
        for i, member in enumerate(visible_members, 1):
            roles = [role.name for role in member.roles if role.name != '@everyone']
            print(f"   {i:2d}. {member.display_name} (@{member.name}) - Roles: {roles}")
        
        # Check available roles
        residency_role_name = config.get('bot.residency_role_name', "Resident '25")
        print(f"\n🏷️ Available Roles:")
        residency_role = None
        for role in guild.roles:
            member_count = "?" if not hasattr(role, 'members') else len(role.members)
            print(f"   - {role.name} (ID: {role.id}, Members: {member_count})")
            if role.name.lower() == residency_role_name.lower():
                residency_role = role
                print(f"     ⭐ FOUND RESIDENCY ROLE!")
        
        if residency_role:
            print(f"\n✅ Residency role found: {residency_role.name}")
            try:
                members = residency_role.members
                print(f"👥 Residency members: {len(members)}")
                for member in members[:10]:  # Show first 10
                    print(f"   - {member.display_name} (@{member.name}, ID: {member.id})")
            except Exception as e:
                print(f"❌ Could not access role members: {e}")
        else:
            print(f"❌ Could not find residency role: '{residency_role_name}'")
        
        # Test member finding manually
        print(f"\n🔍 MANUAL MEMBER SEARCH TEST:")
        test_names = ["Pranav", "pranav", "cassandra", "Cassandra"]
        
        for test_name in test_names:
            found = []
            for member in guild.members:
                if (test_name.lower() in member.display_name.lower() or 
                    test_name.lower() in member.name.lower()):
                    found.append(member)
            
            print(f"Search '{test_name}': {len(found)} matches")
            for match in found[:3]:  # Show first 3
                print(f"   - {match.display_name} (@{match.name}, ID: {match.id})")
        
        await client.close()
    
    try:
        await client.start(token)
    except discord.LoginFailure:
        print("❌ Invalid Discord token")
    except discord.PrivilegedIntentsRequired as e:
        print(f"❌ Privileged intents required: {e}")
        print("💡 To fix this:")
        print("   1. Go to https://discord.com/developers/applications/")
        print("   2. Select your bot application")
        print("   3. Go to the 'Bot' section")
        print("   4. Enable 'Server Members Intent' under 'Privileged Gateway Intents'")
        print("   5. Save changes and restart the bot")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_member_access()) 