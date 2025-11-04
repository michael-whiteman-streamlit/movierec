from __future__ import annotations

import numpy as np
import pandas as pd


def user_embed_from_history(
    item_emb: np.ndarray,
    history_items: list[int],
    learned_user: np.ndarray | None = None,  # e.g., U[u]
    alpha: float = 0.7,  # Weight on history vs learned
    recency_weights: bool = True,
):
    """Compute a user embedding by blending history-based and learned user vectors."""
    # If no history or learned embedding, return None    
    if not history_items and learned_user is None:
        return None

    ue_hist = None
    if history_items:
        # Aggregate embeddings of previously interacted items
        vecs = item_emb[history_items]       
        if recency_weights:
            # Apply linearly increasing weights for recency emphasis
            w = np.linspace(0.5, 1.0, num=len(history_items), dtype=np.float32)
            w = w / (w.sum() + 1e-8)
            ue_hist = (vecs * w[:, None]).sum(axis=0)
        else:
            ue_hist = vecs.mean(axis=0)

    # Combine history and learned representations
    if learned_user is None:
        ue = ue_hist
    elif ue_hist is None:
        ue = learned_user
    else:
        ue = alpha * ue_hist + (1 - alpha) * learned_user

    # Normalize to unit length for cosine similarity use
    ue = ue.astype(np.float32, copy=False)
    n = np.linalg.norm(ue)
    return ue / (n + 1e-12)


def user_seen(train: pd.DataFrame, val: pd.DataFrame):
    """Build a dict mapping each user to all items they've seen in train and val sets."""
    seen={}
    for df in (train, val):
        for u, g in df.groupby('u'):
            s = seen.setdefault(int(u), set())
            s.update(g['i'].tolist())

    return seen
