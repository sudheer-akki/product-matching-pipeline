import io
import numpy as np
from fastapi import Request
from PIL import Image
from core import log_event


async def search_similar_images(request: Request, image_bytes: bytes, top_k = 3):
    #Loading DinoV2 Processor & Faiss Vector DB
    processor = request.app.state.container.processors["dinov2"]
    faiss_index = request.app.state.container.faiss["dinov2"]
    try:
        log_event("info", "Image received for similarity search")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = processor.process(image)["pixel_values"]  # torch.Size([1, 3, 224, 224])
        pixel_values = pixel_values.cpu().numpy().astype(np.float32)

        # Sending Image to Triton Server
        embedding = await request.app.state.batcher_dino.submit({
        "pixel_values": pixel_values
        }) # shape: [1, 768]
        embedding = np.array(embedding, dtype=np.float32)
        log_event("info", "Received Image embedding from Triton", 
        {
            "embedding_shape": embedding.shape
        })
        # Search FAISS Index
        distances, ids = faiss_index.index.search(embedding = embedding, k=top_k)
        log_event("info", "FAISS image search complete", {
            "ids": ids.tolist(),
            "distances": distances.tolist()
        })
        return ids
    
    except Exception as e:
        log_event("error", "Image similarity search failed", {"exception": str(e)})
        return []