# utils/rerank.py
from collections import Counter
from typing import List, Optional

def rerank_results(
    dino_ids: Optional[list] = None,
    caption_bert_ids: Optional[list] = None,
    top_k: int = 5
) -> list:
    score = Counter()
    if dino_ids is not None:
        for i, id_ in enumerate(dino_ids):
            score[id_] += (top_k - i)
    if caption_bert_ids is not None:
        for i, id_ in enumerate(caption_bert_ids):
            score[id_] += (top_k - i)
    ranked_ids = [id_ for id_, _ in score.most_common(top_k)]
    return ranked_ids
