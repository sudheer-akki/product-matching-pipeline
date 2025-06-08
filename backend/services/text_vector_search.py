"""
This module handles operations related to FAISS index search and embedding inference.
"""

from fastapi import Request
from core import log_event
from typing import List
import numpy as np

async def bert_pipeline(request: Request, text: str, top_k: int = 3)-> List[int]:
    """
    Performs vector search using BERT embeddings and FAISS.

    Args:
        request (Request): FastAPI request object to access app state.
        text (str): Input query text.
        top_k (int): Number of top results to return.

    Returns:
        List[int]: Top-k matched document IDs from FAISS.
    """
    try:
        if "bert" not in request.app.state.container.processors:
            raise RuntimeError("BERT tokenizer is not initialized in app.state.processors")

        tokenizer =  request.app.state.container.processors["bert"]
        log_event("info", "Text vector search started", {
            "query": text, 
            "top_k": top_k
        })

        inputs = tokenizer.process(text)
        input_ids = inputs["input_ids"].numpy().astype("int64") # shape: [1, seq_len]
        attention_mask = inputs["attention_mask"].numpy().astype("int64")

        # Inference via Triton
        #embedding = await infer_bert_batch(input_ids=input_ids, attention_mask=attention_mask)
        embedding = None
        # Enqueue the request to the batching system
        try:
            embedding = await request.app.state.batcher_bert.submit(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
        except Exception as e:
            log_event("error", "Bert Embedding Infer failed", {"exception": str(e)})

        if isinstance(embedding, dict):
            if "embedding" not in embedding:
                raise ValueError("Missing 'embedding' key in BERT inference output")
            vector = embedding["embedding"].astype("float32").reshape(1,-1)   # shape: [1, 768]
        else:
            vector = embedding.astype("float32").reshape(1,-1) # shape: [1, 768]

        log_event("info", "Text embedding obtained from Triton", {"embedding_shape": vector.shape})
        ids = search_text_vector(request=request, text_embeddings=vector, top_k= top_k )
        return ids 
    except Exception as e:
        log_event("error", "Text vector search failed", {"query": text, "exception": str(e)})
        return []


def search_text_vector(request: Request, text_embeddings: np.ndarray,  top_k: int = 5)-> List[int]:
    """
    Search FAISS index using BERT embedding vector.

    Args:
        request (Request): FastAPI request object.
        text_embeddings (np.ndarray): Vector to search.
        top_k (int): Number of nearest neighbors to return.

    Returns:
        list[int]: Top-k document IDs.
    """
    faiss_index = request.app.state.container.faiss["bert"]
    distances, ids = faiss_index.index.search(text_embeddings,top_k)
    log_event("info", "FAISS text search complete", {
        "ids": ids.tolist(),
        "distances": distances.tolist()
    })
    return ids[0].tolist()
