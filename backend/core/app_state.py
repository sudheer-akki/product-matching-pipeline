from typing import Optional
from database import FaissIndexHandler
from database import MongoDBClient
from omegaconf import DictConfig


class AppState:
    """
    Central container for application-wide shared instances.
    Initialized once during FastAPI startup via lifespan context.
    """

    mongo: Optional[MongoDBClient] = None
    config: Optional[DictConfig] = None
    faiss: Optional[FaissIndexHandler] = None
