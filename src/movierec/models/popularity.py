from __future__ import annotations
from collections import Counter, defaultdict

import pandas as pd


class PopularityModel:
    """Simple popularity-based recommender ranking items by global frequency."""

    def __init__(self):
        self.item_counts = Counter()
        self.sorted_items = None
        self.user_hist = defaultdict(set)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        """Count item frequencies and record per-user seen items from train/val data."""
        # Train: columns of u, i, ts
        self.item_counts = Counter(train['i'].tolist())
        self.sorted_items = [i for i, _ in self.item_counts.most_common()]

        # User history used to filter seen items
        for df in ([train] + ([val] if val is not None else[])):
            for u, g in df.groupby('u'):
                self.user_hist[u].update(g['i'].tolist())
        
        return self

    def item_popularity(self):
        """Return a dictionary of item popularity counts (for novelty)."""
        return dict(self.item_counts)

    def recommend(self, u:int, k: int, n_items:int):
        """Recommend top-k unseen items for user u based on item popularity."""
        seen = self.user_hist.get(u, set())
        recs = []
        for it in self.sorted_items:
            if it not in seen:
                recs.append(it)
                if len(recs) >= k:
                    break
        
        # If catalogue > sorted items (rare), arbitrarily pad
        if len(recs) < k:
            for it in range(n_items):
                if it not in seen and it not in recs:
                    recs.append(it)
                if len(recs) >= k:
                    break
        
        return recs
