from pymongo import MongoClient
import threading


class MongoDBClient:
    """
    Thread-safe Singleton MongoDB client.
    Provides access to collections and ensures connection health.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, uri, db_name="products"):
        self.uri = uri
        self.db_name = db_name
        self._client = None
        self._connect()

    def _connect(self):
        try:
            from core.custom_log import log_event
            log_event("info", "Connecting to MongoDB", {"uri": self.uri})
            self._client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=2000,
                socketTimeoutMS=2000
            )
            self._client.admin.command("ping")
            log_event("info", "MongoDB connection successful")
        except Exception as e:
            log_event("error", "MongoDB connection failed", {"exception": str(e)})
            raise

    def get_collections(self):
        from core.custom_log import log_event
        try:
            db = self._client[self.db_name]
            meta_data = db["metadata"]
            logs = db["logs"]
            log_event("info", "MongoDB collections retrieved", {"collections": ["metadata", "logs"]})
            return meta_data, logs
        except Exception as e:
            log_event("error", "Failed to get MongoDB collections", {"exception": str(e)})
            raise

    @classmethod
    def get_instance(cls, uri, db_name="products"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(uri, db_name)
        return cls._instance
