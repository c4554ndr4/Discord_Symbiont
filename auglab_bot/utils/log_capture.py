"""
Log capture utility for debug command
"""

import logging
import collections
from typing import List

class LogCapture(logging.Handler):
    """Custom log handler that captures recent log messages for debug command."""
    
    def __init__(self, max_logs: int = 50):
        super().__init__()
        self.max_logs = max_logs
        self.logs = collections.deque(maxlen=max_logs)
        
    def emit(self, record):
        """Capture log record."""
        try:
            msg = self.format(record)
            self.logs.append({
                'timestamp': record.created,
                'level': record.levelname,
                'message': msg,
                'module': record.name
            })
        except Exception:
            pass  # Don't let log capture break logging
            
    def get_recent_logs(self, count: int = 20, level_filter: str = None) -> List[dict]:
        """Get recent log messages."""
        logs = list(self.logs)
        
        # Filter by level if specified
        if level_filter:
            logs = [log for log in logs if log['level'] == level_filter.upper()]
            
        # Return most recent
        return logs[-count:] if count else logs
    
    def get_error_logs(self, count: int = 10) -> List[dict]:
        """Get recent error messages."""
        return self.get_recent_logs(count, 'ERROR')

# Global instance
log_capture = LogCapture()

def setup_log_capture():
    """Set up log capture for the root logger."""
    root_logger = logging.getLogger()
    
    # Check if already added
    for handler in root_logger.handlers:
        if isinstance(handler, LogCapture):
            return handler
            
    # Add log capture handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    log_capture.setFormatter(formatter)
    root_logger.addHandler(log_capture)
    
    return log_capture