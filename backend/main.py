import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from core import AppState, load_config, log_event, initialize_log_event
from database import MongoDBClient, FaissIndexHandler
from models.bert_model import BERTTextProcessor
from models.dinov2_model import DINOv2Preprocessor
from routes.api_router import router as api_router
from batching import InferenceBatcher
from services import (
    infer_bert,
    infer_dinov2,
    infer_llava_bert
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles FastAPI startup and shutdown lifecycle.
    Initializes config, MongoDB, logging, and FAISS index.
    """
    try:
        print("[INFO] Startup initiated")

        # Create container for shared state
        app.state.container = AppState()

        # Load application configuration
        config = load_config()
        app.state.container.config = config

        print("MongoDB Intialization Started")
        # Initialize MongoDB client
        mongo_uri = config.mongo.uri
        mongo_client = MongoDBClient.get_instance(uri=mongo_uri)
        app.state.container.mongo = mongo_client
        print("MongoDB Client Registered")

        # Initialize logging system (requires MongoDB)
        initialize_log_event(mongo_client)

        faiss_indexes = {}
        for model_name, faiss_cfg in config.faiss.items():
            faiss_indexes[model_name] = FaissIndexHandler(
                saved_index_path=faiss_cfg.path,
                dimension=faiss_cfg.dimension
            )
            log_event("info", f"FAISS index for '{model_name}' loaded")
        app.state.container.faiss = faiss_indexes

        processors = {
            "dinov2": DINOv2Preprocessor(model_name=config.models.dinov2.name),
            "bert": BERTTextProcessor(model_name=config.models.bert.name)
        }
        app.state.container.processors = processors
        log_event("info", "Model encoders loaded: DINOv2 and BERT")

        app.state.batcher_bert = InferenceBatcher(
        max_batch_size=8,
        max_wait_time=0.05,  # or tune based on load
        infer_fn=infer_bert  # must accept List of inputs
        )
        asyncio.create_task(app.state.batcher_bert._batch_worker())

        app.state.batcher_dino = InferenceBatcher(
            max_batch_size=8,
            max_wait_time=0.05,
            infer_fn=infer_dinov2  # define similar function
        )
        asyncio.create_task(app.state.batcher_dino._batch_worker())

        app.state.batcher_llava_bert = InferenceBatcher(
            max_batch_size=8,
            max_wait_time=0.05,
            infer_fn=infer_llava_bert
        )
        asyncio.create_task(app.state.batcher_llava_bert._batch_worker())

        log_event("info", "Application startup complete")
        yield

    except Exception as e:
        print(f"[ERROR] Startup failed: {e}")
        try:
            log_event("error", "Startup failure", {"exception": str(e)})
        except Exception:
            pass
        raise

app = FastAPI(lifespan=lifespan, title="Product Search API")

@app.api_route("/search/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}


# CORS: allow frontend (Gradio or other UI) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"], #["http://localhost:7860"], 
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# Include all routes
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    #from core import log_event
    try:
        print("info", "Starting FastAPI server")
        uvicorn.run("main:app", host="0.0.0.0", port=6000, reload=False)
    except Exception as e:
        print("error", "Uvicorn server failed to start", {"exception": str(e)})
