# 🌐 Guild Management Guide

## How to Check Which Servers Your Bot Is In

Use the new `!guilds` command in Discord to see all servers where your bot is currently active:

```
!guilds
```

This will show you:
- Server names and IDs
- Member counts
- Bot permissions
- Which server is the "target guild" (where the bot is most active)
- Current restriction status

## How to Restrict Bot to Specific Servers

To prevent the bot from being active in unwanted servers like "churn observatory":

### Method 1: Configuration File (Recommended)

1. **Edit `config.json`:**
   ```json
   {
     "bot": {
       "prefix": "!",
       "target_guild_id": 0,
       "residency_role_name": "Resident '25",
       "database_path": "./auglab_bot.db",
       "debug": false,
       "allowed_guilds": [123456789012345678],
       "restricted_mode": true
     }
   }
   ```

2. **Set the configuration:**
   - `allowed_guilds`: List of Discord server IDs where the bot should be active
   - `restricted_mode`: Set to `true` to enable restrictions

3. **Get Server IDs:**
   - Use `!guilds` command to see server IDs
   - Or enable Developer Mode in Discord → Right-click server → Copy Server ID

4. **Restart the bot:**
   ```bash
   pkill -f "python discord_bot_auglab.py"
   python discord_bot_auglab.py &
   ```

### Method 2: Remove Bot from Unwanted Servers

1. **Go to the unwanted Discord server**
2. **Server Settings → Members → Find your bot**
3. **Click "Kick" or "Ban"**
4. **Or use server permissions to restrict bot access**

## Example Configuration

To restrict the bot to only work in your main Aug Lab server:

```json
{
  "bot": {
    "allowed_guilds": [1234567890123456789],
    "restricted_mode": true,
    "target_guild_id": 1234567890123456789
  }
}
```

## How It Works

- **`restricted_mode: false`** (default): Bot works in all servers
- **`restricted_mode: true`**: Bot only processes messages in allowed servers
- **`allowed_guilds: []`**: List of server IDs where bot should be active
- **Bot still shows up in other servers** but won't respond to messages

## Verification

After making changes:
1. Use `!guilds` to verify settings
2. Test in different servers to confirm restrictions work
3. Bot should only respond in allowed servers

## Notes

- The bot will still appear online in restricted servers
- It just won't process messages or respond
- Commands like `!guilds` work regardless of restrictions
- DMs are not affected by guild restrictions 