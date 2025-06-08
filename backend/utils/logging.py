from datetime import datetime

def safe_log_event(level, message, metadata=None):
    try:
        from core import log_event
        log_event(level, message, metadata)
    except Exception as e:
        timestamp = datetime.utcnow().isoformat()
        print(f"[{timestamp}] {level.upper()}: {message}")
        if metadata:
            print("  ↳", metadata)
        print(f" Fallback logging used. Reason: {e}")
