from cachetools import LRUCache
from typing import List, Tuple
from fastapi import Request
from core import log_event

metadata_cache = LRUCache(maxsize=1000)


def get_cached_metadata(ids: List[int]) -> Tuple[List[dict], List[int]]:
    """
    Retrieves cached metadata for the given list of numeric IDs.

    Args:
        ids (List[int]): List of product numeric IDs to retrieve.

    Returns:
        Tuple[List[Dict], List[int]]:
            - A list of metadata documents found in cache.
            - A list of IDs that were not found in the cache.
    """
    cached_docs = []
    missing_ids = []

    for id_ in ids:
        if id_ in metadata_cache:
            cached_docs.append(metadata_cache[id_])
        else:
            missing_ids.append(id_)
    return cached_docs, missing_ids


def fetch_metadata(request: Request, numeric_ids: List[int]) -> List[dict]:
    """
    Fetches metadata documents from MongoDB based on numeric IDs.

    Args:
        request (Request): FastAPI request to access app state.
        numeric_ids (List[int]): FAISS result IDs.

    Returns:
        List[dict]: Ordered list of metadata documents.
    """
    try:

        db = request.app.state.container.mongo._client["products"]
        meta_data = db["metadata"]

        # Check cache first instead of hitting MongoDB.
        cached_docs, missing_ids = get_cached_metadata(numeric_ids)
        db_results = []

        if missing_ids:
            cursor = meta_data.find(
                {"numeric_id": {"$in": missing_ids}},
                {"numeric_id": 1, "image": 1, "metadata": 1, "_id": 0}
            )
            db_results = cursor.to_list(length=len(missing_ids))

            # Storing in cache
            for doc in db_results:
                metadata_cache[doc["numeric_id"]] = doc

        # Combine cached results
        all_docs = cached_docs + db_results

        # Preserve original FAISS order
        id_to_doc = {doc["numeric_id"]: doc for doc in all_docs}
        ordered_results = [id_to_doc[i] for i in numeric_ids if i in id_to_doc]

        log_event("info", "Metadata fetched from MongoDB", {
            "requested_ids": numeric_ids,
            "returned_count": len(ordered_results)
        })
        return ordered_results

    except Exception as e:
        log_event("error", "fetch_metadata failed", {
            "exception": str(e),
            "ids": numeric_ids
        })
        return []