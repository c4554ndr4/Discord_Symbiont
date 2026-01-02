#!/usr/bin/env python3
"""
Create member mapping from available sources (messages, memory, etc.)
This works around the missing Server Members Intent limitation.
"""

import os
import sys
import sqlite3
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

from auglab_bot.config.manager import ConfigManager

def extract_members_from_database():
    """Extract member information from the bot's database."""
    config = ConfigManager()
    db_path = config.get('bot.database_path', './auglab_bot.db')
    
    print(f"🔍 Analyzing database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return {}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check available tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"📋 Available tables: {tables}")
    
    members = {}
    
    # Extract from messages table
    if 'messages' in tables:
        print("\n💬 Extracting from messages...")
        cursor.execute("""
            SELECT DISTINCT author_id, author_name, guild_id 
            FROM messages 
            WHERE author_id IS NOT NULL AND author_name IS NOT NULL
            ORDER BY timestamp DESC
        """)
        
        for row in cursor.fetchall():
            author_id, author_name, guild_id = row
            if author_id not in members:
                members[author_id] = {
                    'user_id': author_id,
                    'username': author_name,
                    'display_name': author_name,
                    'guild_id': guild_id,
                    'source': 'messages'
                }
    
    # Extract from memory table
    if 'memory' in tables:
        print("🧠 Extracting from memory...")
        cursor.execute("""
            SELECT DISTINCT user_id, metadata, content 
            FROM memory 
            WHERE user_id IS NOT NULL AND user_id != 0
        """)
        
        for row in cursor.fetchall():
            user_id, metadata, content = row
            if user_id not in members:
                # Try to extract name from content or metadata
                name = f"User_{user_id}"
                if metadata:
                    try:
                        meta = json.loads(metadata)
                        if 'user_name' in meta:
                            name = meta['user_name']
                    except:
                        pass
                
                members[user_id] = {
                    'user_id': user_id,
                    'username': name,
                    'display_name': name,
                    'guild_id': None,
                    'source': 'memory'
                }
    
    # Extract from cost_tracker if available
    if 'cost_entries' in tables:
        print("💰 Extracting from cost tracker...")
        cursor.execute("""
            SELECT DISTINCT user_id 
            FROM cost_entries 
            WHERE user_id IS NOT NULL AND user_id != 0
        """)
        
        for row in cursor.fetchall():
            user_id = row[0]
            if user_id not in members:
                members[user_id] = {
                    'user_id': user_id,
                    'username': f"User_{user_id}",
                    'display_name': f"User_{user_id}",
                    'guild_id': None,
                    'source': 'cost_tracker'
                }
    
    conn.close()
    return members

def create_member_mapping_file():
    """Create a member mapping file for manual editing."""
    
    # Extract from database
    members = extract_members_from_database()
    
    print(f"\n📊 Found {len(members)} unique members from database")
    
    # Known members from logs (you can add more here)
    known_members = {
        1377061736154140732: {
            'user_id': 1377061736154140732,
            'username': 'cassandra',
            'display_name': 'Cassandra',
            'possible_names': ['cassandra', 'Cassandra'],
            'source': 'logs'
        }
        # Add more known members here based on your knowledge
    }
    
    # Merge with known members
    for user_id, info in known_members.items():
        members[user_id] = info
    
    # Create the mapping file
    mapping_file = 'member_mapping.json'
    
    member_list = []
    for user_id, info in members.items():
        member_data = {
            'user_id': int(user_id),
            'username': info.get('username', f'User_{user_id}'),
            'display_name': info.get('display_name', f'User_{user_id}'),
            'possible_names': info.get('possible_names', [info.get('username', f'User_{user_id}')]),
            'source': info.get('source', 'unknown'),
            'is_resident': False  # You'll need to manually mark these
        }
        member_list.append(member_data)
    
    # Sort by user_id
    member_list.sort(key=lambda x: x['user_id'])
    
    with open(mapping_file, 'w') as f:
        json.dump(member_list, f, indent=2)
    
    print(f"\n✅ Created {mapping_file} with {len(member_list)} members")
    print(f"📝 Please edit this file to:")
    print(f"   1. Set 'is_resident': true for Residency '25 members")
    print(f"   2. Add correct usernames/display names")
    print(f"   3. Add possible name variations to 'possible_names'")
    
    # Print some examples
    print(f"\n📋 Sample entries:")
    for member in member_list[:5]:
        print(f"   - {member['display_name']} (ID: {member['user_id']}, Source: {member['source']})")
    
    return mapping_file

def update_find_member_function():
    """Update the find_member_by_name function to use the mapping file."""
    
    mapping_file = 'member_mapping.json'
    if not os.path.exists(mapping_file):
        print(f"❌ {mapping_file} not found. Run create_member_mapping_file() first.")
        return
    
    # Read the mapping
    with open(mapping_file, 'r') as f:
        members = json.load(f)
    
    # Generate the function code
    function_code = '''
def find_member_by_name_with_mapping(self, name: str) -> Dict[str, Any]:
    """Find member by name using static mapping (fallback for missing Members Intent)."""
    
    # Load member mapping
    mapping_file = "member_mapping.json"
    if not os.path.exists(mapping_file):
        return {"success": False, "error": "Member mapping file not found"}
    
    try:
        with open(mapping_file, 'r') as f:
            members = json.load(f)
        
        # Search for member
        name_lower = name.lower()
        
        for member in members:
            # Skip non-residents if we're looking for residents
            if not member.get('is_resident', False):
                continue
                
            # Check all possible name variations
            possible_names = member.get('possible_names', [])
            possible_names.extend([member.get('username', ''), member.get('display_name', '')])
            
            for possible_name in possible_names:
                if possible_name and name_lower in possible_name.lower():
                    return {
                        "success": True,
                        "user_id": member['user_id'],
                        "username": member['username'],
                        "display_name": member['display_name']
                    }
        
        return {"success": False, "error": f"No resident member found with name '{name}'"}
        
    except Exception as e:
        return {"success": False, "error": f"Error reading member mapping: {str(e)}"}
'''
    
    print("📝 Function code generated. You can add this to your bot's function implementations.")
    print("💡 Alternatively, I can patch the existing function to use this mapping as a fallback.")

if __name__ == "__main__":
    print("🔧 Creating member mapping for Discord bot...")
    print("=" * 60)
    
    mapping_file = create_member_mapping_file()
    
    print(f"\n🎯 Next steps:")
    print(f"1. Edit {mapping_file} manually")
    print(f"2. Set is_resident: true for Residency '25 members") 
    print(f"3. Add proper usernames/display names")
    print(f"4. Add name variations (nicknames, first names, etc.)")
    print(f"5. Run the bot with updated find_member_by_name function")
    
    print(f"\n🚀 Or enable Server Members Intent for automatic discovery!") 