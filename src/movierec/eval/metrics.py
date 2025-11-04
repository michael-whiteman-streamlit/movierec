from __future__ import annotations
import math
from typing import List

import numpy as np


def coverage_at_k(all_ranked_topk, n_items: int):
    """Coverage@K: proportion of catalog exposed across users' top-K lists."""
    exposed = set()
    for arr in all_ranked_topk:
        exposed.update(arr)
    return len(exposed) / n_items if n_items > 0 else 0.0


def hitrate_at_k(ranked_items, ground_truth_set, k=10):
    """HitRate@K: 1 if any ground-truth item appears in top-K, else 0."""
    return 1.0 if any((it in ground_truth_set) for it in ranked_items[:k]) else 0.0


def ndcg_at_k(ranked_items, ground_truth_set, k=10):
    """NDCG@K: normalized DCG of `ranked_items[:k]` versus `ground_truth_set`."""
    hits = np.array([1 if it in ground_truth_set else 0 for it in ranked_items[:k]], dtype=float)
    dcg = _dcg(hits)
    ideal = _dcg(np.sort(hits)[::-1])
    return float(dcg / ideal) if ideal > 0 else 0.0


def novelty_at_k(topk_lists, item_popularity, k=10):
    """Compute mean novelty across users’ top-k recommendations using inverse item popularity."""
    novs=[]
    for items in topk_lists:
        vals = []
        for it in items[:k]:
            p = item_popularity.get(it, 1)
            vals.append(-math.log(p))
        if vals:
            novs.append(sum(vals)/len(vals))
    return float(np.mean(novs)) if novs else 0.0


def novelty_bits_for_list(items: List[int], pop_counts: np.ndarray) -> float:
    """Compute average novelty in bits for a list based on item popularity counts."""
    total = int(pop_counts.sum())
    if total <= 0 or not items:
        return 0.0
    p = np.clip(pop_counts[np.array(items, dtype=int)] / total, 1e-12, 1.0)
    return float(np.mean(-np.log2(p)))


def recall_at_k(ranked_items, ground_truth_set, k=10):
    """Recall@K: fraction of `ground_truth_set` retrieved in top-K."""
    if not ground_truth_set:
        return 0.0
    hits = sum((it in ground_truth_set) for it in ranked_items[:k])
    return hits / len(ground_truth_set)


def _dcg(rel):
    """Compute DCG for a ranked binary relevance vector `rel` (1 = relevant)."""
    idx = np.arange(1, len(rel) + 1)
    return np.sum(rel / np.log2(idx + 1))
