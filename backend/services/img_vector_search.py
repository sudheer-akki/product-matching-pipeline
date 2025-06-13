import io
import numpy as np
from fastapi import Request
from PIL import Image
from core import log_event
from .triton_client import infer_dinov2


async def search_similar_images(request: Request, image_bytes: bytes, top_k = 5):
    #Loading DinoV2 Processor & Faiss Vector DB
    processor = request.app.state.container.processors["dinov2"]
    faiss_index = request.app.state.container.faiss["dinov2"]
    try:
        log_event("info", "Image received for similarity search")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = processor.process(image)["pixel_values"]  # torch.Size([1, 3, 224, 224])
        pixel_values = pixel_values.cpu().numpy().astype(np.float32)

        # Sending Image to Triton Server
        embedding = await infer_dinov2( #await request.app.state.batcher_dino.submit({
        {
           "pixel_values": pixel_values
        }
        ) # shape: [1, 768]

        # embedding = await request.app.state.batcher_dino.submit(
        # {
        #     "pixel_values": pixel_values
        # }
        # )
        embedding = embedding["embedding"]    
        shape = embedding["embedding_shape"]
        #if isinstance(vector, list):
        #    vector = np.asarray(vector, dtype=np.float32)
        #embedding = np.asarray(embedding, dtype=np.float32)

        log_event("info", "Received Image embedding from Triton", {"embedding_shape": shape})
        # Search FAISS Index
        ids, distances = faiss_index._search_index(embedding=embedding,top_k=top_k, name = "dinov2")
        log_event("info", "FAISS image search complete", {
        "ids": ids,
        "distances":distances
        })
        return ids
    
    except Exception as e:
        log_event("error", "Image similarity search failed", {"exception": str(e)})
        return []