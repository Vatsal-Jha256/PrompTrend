from datetime import datetime

def format_timestamp(dt: datetime) -> str:
    return dt.isoformat()

def validate_chat_history(chat_history):
    """Validate chat history format and content"""
    if not chat_history.messages:
        return False
    return all(
        isinstance(m.content, str) and len(m.content.strip()) > 0
        for m in chat_history.messages
    )