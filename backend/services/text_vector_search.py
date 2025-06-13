
from fastapi import Request
from core import log_event
from typing import List
import numpy as np
from .triton_client import infer_bert

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
        embedding = None
        # Enqueue the request to the batching system
        try:
            embedding = await infer_bert ( 
            {
               "input_ids": input_ids,
               "attention_mask": attention_mask
            })

            # embedding = await request.app.state.batcher_bert.submit(
            # {
            #     "input_ids": input_ids,
            #     "attention_mask": attention_mask
            # })
        except Exception as e:
            log_event("error", "Bert Embedding Infer failed", {"exception": str(e)})
        vector = embedding["embedding"]    
        shape = embedding["embedding_shape"]
        if isinstance(vector, list):
            vector = np.asarray(vector, dtype=np.float32)
        log_event("info", "Text embedding obtained from Triton", {"embedding_shape": shape})
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
    ids, distances = faiss_index._search_index(embedding=text_embeddings,top_k=top_k, name = "bert")
    log_event("info", "FAISS text search complete", {
        "ids": ids,
        "distances": distances
    })
    return ids
