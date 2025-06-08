# utils/rerank.py
from collections import Counter
from typing import List, Optional

def rerank_results(
    dino_ids: Optional[List[str]] = None,
    caption_bert_ids: Optional[List[str]] = None,
    text_match_ids: Optional[List[str]] = None,
    top_k: int = 5,
    weights: dict = None
) -> List[str]:
    weights = weights or {"dino": 1.0, "caption": 1.0, "text": 1.0}
    score = Counter()

    if dino_ids is not None:
        for i, id_ in enumerate(dino_ids):
            score[id_] += weights["dino"] * (top_k - i)

    if caption_bert_ids is not None:
        for i, id_ in enumerate(caption_bert_ids):
            score[id_] += weights["caption"] * (top_k - i)

    if text_match_ids is not None:
        for i, id_ in enumerate(text_match_ids):
            score[id_] += weights["text"] * (top_k - i)


    ranked_ids = [id_ for id_, _ in score.most_common(top_k)]
    return ranked_ids
