from datetime import datetime
from pymongo.collection import Collection
from typing import Optional
import numpy as np

# Flags to control output
LOG_TO_CONSOLE: bool = True
LOG_TO_MONGODB: bool = False

# Internal Mongo collection reference
_logs_collection: Optional[Collection] = None


def initialize_log_event(mongo_client) -> None:
    """
    Initializes the log system by binding the MongoDB logs collection.
    This must be called after MongoDB is connected.

    Args:
        mongo_client: Initialized MongoDBClient instance.
    """
    global LOG_TO_MONGODB, _logs_collection

    try:
        db = mongo_client._client["products"]
        _logs_collection = db["logs"]
        LOG_TO_MONGODB = True

        if LOG_TO_CONSOLE:
            print("[INFO] Log event system initialized with MongoDB")
    except Exception as e:
        _logs_collection = None
        LOG_TO_MONGODB = False

        if LOG_TO_CONSOLE:
            print(f"[WARNING] Failed to initialize log event system with MongoDB: {e}")


def log_event(level: str, message: str, metadata: Optional[dict] = None) -> None:
    """
    Logs an event to console and optionally MongoDB.

    Args:
        level (str): Logging level, e.g., 'info', 'error', etc.
        message (str): Message to be logged.
        metadata (dict, optional): Additional metadata for context.
    """
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    if metadata:
        metadata = {k: make_serializable(v) for k, v in metadata.items()}
    else:
        metadata = {}

    log_entry = {
        "timestamp": datetime.utcnow(),
        "level": level.upper(),
        "message": message,
        "metadata": metadata or {}
    }


    if LOG_TO_CONSOLE:
        print(f"[{log_entry['timestamp']}] {log_entry['level']}: {log_entry['message']}")
        if log_entry["metadata"]:
            print("  ↳ Metadata:", log_entry["metadata"])

    if LOG_TO_MONGODB and _logs_collection is not None:
        try:
            _logs_collection.insert_one(log_entry)
        except Exception as e:
            if LOG_TO_CONSOLE:
                print(f"[ERROR] Logging to MongoDB failed: {e}")



# from datetime import datetime

# LOG_TO_CONSOLE = True
# LOG_TO_MONGODB = False  # turn on later

# def log_event(level: str, message: str, metadata: dict = None):
#     log_entry = {
#         "timestamp": datetime.utcnow(),
#         "level": level.upper(),
#         "message": message,
#         "metadata": metadata or {}
#     }

#     if LOG_TO_CONSOLE:
#         print(f"[{log_entry['timestamp']}] {log_entry['level']}: {log_entry['message']}")
#         if log_entry["metadata"]:
#             print("  ↳ Metadata:", log_entry["metadata"])

#     if LOG_TO_MONGODB:
#         try:
#             from database import get_collections
#             _, logs_collection = get_collections()
#             logs_collection.insert_one(log_entry)
#         except Exception as e:
#             if LOG_TO_CONSOLE:
#                 print(f"Logging to MongoDB failed: {e}")
