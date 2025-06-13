from fastapi import APIRouter, Form, Request
from services import bert_pipeline, fetch_metadata
from core import log_event
import asyncio

router = APIRouter()

@router.post("/search/text")
async def search_text(request: Request, query: str = Form(...)):
    request_id = request.headers.get("x-request-id","unknown")
    log_event("info", "Text search request received", {
        "query": query,
        "request_id": request_id
    })
    try:
        # BERT + FAISS Search Pipeline
        bert_search = asyncio.create_task(bert_pipeline(request,text=query, top_k=5))
        #numeric_ids = await bert_pipeline(request,text=query, top_k=5)
    
        numeric_ids = await asyncio.wait_for(bert_search, timeout=5)
        log_event("info", "BERT + FAISS search complete", {
            "result_count": len(numeric_ids),
            "result": numeric_ids,
            "request_id": request_id
        })

        if not numeric_ids:
            return {"query": query, "results": []}
        
        if numeric_ids and isinstance(numeric_ids[0], list):
            numeric_ids = numeric_ids[0][0]
        # Fetch metadata from MongoDB using Numeric IDs
        results = fetch_metadata(request, numeric_ids = numeric_ids)
        log_event("info", "Metadata fetched for text search", {
            "query": query,
            "result_count": len(results),
            "request_id": request_id
        })
        return {"query": query, "results": results}

    except Exception as e:
        log_event("error", "Text search failed", {
            "query": query,
            "exception": str(e),
            "request_id": request_id
        })
        return {"error": str(e)}
