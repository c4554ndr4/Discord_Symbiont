"""
Cost tracking for Augmentation Lab Bot
"""

import os
import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

@dataclass
class CostEntry:
    """Represents a cost entry for tracking compute usage."""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    action: str
    user_id: int
    channel_id: int


class CostTracker:
    """Tracks and manages compute costs with budget limits."""
    
    def __init__(self, budget: float, db_path: str):
        self.budget = budget
        self.db_path = db_path
        self.init_database()
        
        # Cost rates per 1M tokens (from environment)
        self.rates = {
            'haiku': {
                'input': float(os.getenv('HAIKU_INPUT_COST', '0.25')),
                'output': float(os.getenv('HAIKU_OUTPUT_COST', '1.25'))
            },
            'sonnet': {
                'input': float(os.getenv('SONNET_INPUT_COST', '3.0')),
                'output': float(os.getenv('SONNET_OUTPUT_COST', '15.0'))
            },
            'gpt4o': {
                'input': float(os.getenv('GPT4O_INPUT_COST', '2.5')),
                'output': float(os.getenv('GPT4O_OUTPUT_COST', '10.0'))
            }
        }
        
    def init_database(self):
        """Initialize the SQLite database for cost tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cost_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    action TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS member_cache (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    roles TEXT NOT NULL,
                    joined_at TEXT,
                    last_updated TEXT NOT NULL
                )
            ''')
            
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a given model and token usage."""
        if model not in self.rates:
            logger.warning(f"Unknown model for cost calculation: {model}")
            return 0.0
            
        input_cost = (input_tokens / 1_000_000) * self.rates[model]['input']
        output_cost = (output_tokens / 1_000_000) * self.rates[model]['output']
        return input_cost + output_cost
    
    def can_afford(self, model: str, estimated_tokens: int) -> bool:
        """Check if we can afford the estimated cost."""
        current_spending = self.get_total_spending()
        estimated_cost = self.calculate_cost(model, estimated_tokens, estimated_tokens)
        return (current_spending + estimated_cost) <= self.budget
    
    def record_usage(self, entry: CostEntry):
        """Record a cost entry in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO cost_entries 
                (timestamp, model, input_tokens, output_tokens, cost, action, user_id, channel_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.timestamp.isoformat(),
                entry.model,
                entry.input_tokens,
                entry.output_tokens,
                entry.cost,
                entry.action,
                entry.user_id,
                entry.channel_id
            ))
    
    def get_total_spending(self) -> float:
        """Get total spending so far."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT SUM(cost) FROM cost_entries')
            result = cursor.fetchone()[0]
            return result if result else 0.0
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return self.budget - self.get_total_spending()
    
    def get_spending_report(self) -> Dict[str, Any]:
        """Get detailed spending report."""
        with sqlite3.connect(self.db_path) as conn:
            # Total by model
            cursor = conn.execute('''
                SELECT model, SUM(cost), SUM(input_tokens), SUM(output_tokens), COUNT(*)
                FROM cost_entries 
                GROUP BY model
            ''')
            by_model = {row[0]: {
                'cost': row[1], 'input_tokens': row[2], 
                'output_tokens': row[3], 'requests': row[4]
            } for row in cursor.fetchall()}
            
            # Recent entries
            cursor = conn.execute('''
                SELECT timestamp, model, cost, action
                FROM cost_entries 
                ORDER BY timestamp DESC LIMIT 10
            ''')
            recent = [{'timestamp': row[0], 'model': row[1], 'cost': row[2], 'action': row[3]} 
                     for row in cursor.fetchall()]
            
            return {
                'total_spent': self.get_total_spending(),
                'remaining': self.get_remaining_budget(),
                'budget': self.budget,
                'by_model': by_model,
                'recent_entries': recent
            } 