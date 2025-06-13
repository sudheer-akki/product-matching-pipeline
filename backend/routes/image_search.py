import asyncio
from fastapi import Request, APIRouter, UploadFile, File
from services import (
    search_similar_images,
    search_text_vector,
    fetch_metadata,
    llava_bert_pipeline  
)
from core import log_event
from utils import rerank_results

router = APIRouter()


@router.post("/search/image")
async def search_image(request: Request, image: UploadFile = File(...), top_k: int = 3):
    request_id = request.headers.get("x-request-id", "unknown")
    log_event("info", "Image search request received", {
        "filename": image.filename,
        "top_k": top_k,
        "request_id": request_id
    })

    try:
        image_bytes = await image.read()

        log_event("info", "Sending Image to DinoV2", {"request_id": request_id})
        dino_task = asyncio.create_task(search_similar_images(request, image_bytes, top_k= top_k))

        log_event("info", "Sending Image to Llava -> Bert Pipeline", {"request_id": request_id})
        llava_bert_response = asyncio.create_task(llava_bert_pipeline(
            request = request,
            image_bytes = image_bytes,
            prompt = "Describe the Gender, clothing and accessories in the image",
            max_tokens=128,
            temperature=0.2,
            top_k=1,
            freq_penalty=0.3,
            seed=42
        ))
        try:
            caption, bert_embedding = await asyncio.wait_for(llava_bert_response, timeout=5)
            log_event("info", "Generated Llava->Bert Embeddings", 
                    {"request_id": request_id})
        except asyncio.TimeoutError:
            caption = None
            log_event("warn", "Generating Llava->Bert Embeddings timed out", {"request_id": request_id})

        caption_bert_ids = []
        if caption and caption.strip():
            log_event("info", "Caption generated", {"request_id": request_id})
            caption_bert_ids = search_text_vector(request,text_embeddings=bert_embedding, top_k=top_k)
            log_event("info", "Bert search complete", {"caption_bert_ids": caption_bert_ids, "request_id": request_id})
            
        dino_ids = await dino_task
        if dino_ids and isinstance(dino_ids[0], list):
            dino_ids = dino_ids[0]
        log_event("info", "DINO search complete", {"dino_ids": dino_ids, "request_id": request_id})
        # Combine and rerank
        log_event("info", "Sending to reranking", {
            "result_count": len(dino_ids) + len(caption_bert_ids),
            "request_id": request_id
        })

        ranked_ids = rerank_results(
            dino_ids=dino_ids,
            caption_bert_ids=caption_bert_ids,
            top_k=top_k
        )

        log_event("info", "Reranked results", {
            "result_count": len(ranked_ids),
            "ranked_ids": ranked_ids,
            "request_id": request_id
        })

        #Fetching Images from MongoDB using Numeric IDs
        results = fetch_metadata(request, numeric_ids = ranked_ids)
        log_event("info", "Metadata fetched", 
                {"result_count": len(results), 
                "request_id": request_id})

        return {
            "caption": caption or "Caption unavailable.",
            "results": results
        }

    except Exception as e:
        log_event("error", "Image search failed", {"exception": str(e), "request_id": request_id})
        #raise HTTPException(status_code=500, detail="Image search failed")
        #return {"error": str(e)}