from __future__ import annotations
import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from movierec.models.popularity import PopularityModel
from movierec.eval.metrics import (
    coverage_at_k,
    hitrate_at_k,
    ndcg_at_k,
    novelty_bits_for_list,
    recall_at_k,
)


def build_pop_order(train: pd.DataFrame, n_items: int) -> Tuple[np.ndarray, np.ndarray]:
    counts = train['i'].value_counts().reindex(range(n_items)).fillna(0).astype(np.int64).to_numpy()
    order  = np.argsort(-counts, kind='mergesort')
    return order, counts


def build_seen_index(train: pd.DataFrame, val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Create a compact index over seen interactions:
      seen_u: shape [M], sorted by user id
      seen_i: shape [M], aligned items
      offset: dict user_id -> start index in seen arrays (use counts to get length)
    """
    both = pd.concat([train[['u','i']], val[['u','i']]], axis=0, ignore_index=True)
    both = both.sort_values(['u', 'i'], kind='mergesort').drop_duplicates(['u','i'])  # de-dup just in case
    seen_u = both['u'].to_numpy(np.int64, copy=False)
    seen_i = both['i'].to_numpy(np.int64, copy=False)

    # Compute first occurrence index per user
    unique_users, first_idx = np.unique(seen_u, return_index=True)
    offset = {int(u): int(idx) for u, idx in zip(unique_users, first_idx)}
    return seen_u, seen_i, offset


def get_seen_for_user(u: int, seen_u: np.ndarray, seen_i: np.ndarray, offset: Dict[int, int]) -> np.ndarray:
    """Return the slice of seen_i for user u (may be empty)."""
    if u not in offset:
        return np.empty(0, dtype=np.int64)
    start = offset[u]
    # end is next user's start or array end
    # find next index using np.searchsorted over seen_u
    end = np.searchsorted(seen_u, u, side='right', sorter=None)
    return seen_i[start:end]


def load_splits(splits_dir: pathlib.Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(splits_dir / 'train.parquet')
    val = pd.read_parquet(splits_dir / 'val.parquet')
    test = pd.read_parquet(splits_dir / 'test.parquet')
    
    # Enforce compact dtypes
    for df in (train, val, test):
        if 'u' in df.columns: df['u'] = df['u'].astype(np.int64)
        if 'i' in df.columns: df['i'] = df['i'].astype(np.int64)
    return train, val, test


def per_user_seen(train: pd.DataFrame, val: pd.DataFrame) -> Dict[int, set]:
    seen = {}
    both = pd.concat([train[['u','i']], val[['u','i']]], axis=0)
    for u, g in both.groupby('u', sort=False):
        seen[int(u)] = set(map(int, g['i'].tolist()))
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_dir', required=True, help='artifacts/<dataset>/splits')
    parser.add_argument('--items_path', required=True, help='artifacts/<dataset>/items.parquet')
    parser.add_argument('--out_json', required=True, help='where to save the metrics json')
    parser.add_argument('--ks', default='10,20', help='comma-separated cutoffs, e.g. 10,20,50')
    args = parser.parse_args()

    splits_dir = pathlib.Path(args.splits_dir)
    items_path = pathlib.Path(args.items_path)

    # Load splits created using make_splits.py
    train, val, test = load_splits(splits_dir)

    # Infer catalog size from items, fallback to splits
    items = pd.read_parquet(items_path)
    if 'i' in items.columns:
        items = items.set_index('i').sort_index()
    n_items = int(items.index.max()) + 1 if len(items) else int(train['i'].max()) + 1

    # Streamining-friendly indices
    seen_u, seen_i, offset = build_seen_index(train, val)
    pop_order, pop_counts = build_pop_order(train, n_items)

    # Ground truth as arrays (leave-one-out friendly)
    test_u = test['u'].to_numpy(np.int64, copy=False)
    test_i = test['i'].to_numpy(np.int64, copy=False)

    # per-user slices in test
    u_unique, u_first = np.unique(test_u, return_index=True)
    u_first_sorted = np.argsort(u_first)
    u_order = u_unique[u_first_sorted]  # users in encounter order

    Ks = [int(x) for x in args.ks.split(',') if x.strip()]
    k_star = max(Ks) if Ks else 20

    # Accumulators
    results = {K: {'rec': 0.0, 'ndcg': 0.0, 'hit': 0.0, 'cov_set': set(), 'nov': 0.0} for K in Ks}
    n_users = 0

    for u in u_order:
        n_users += 1
        
        # Ground truth set for user u
        mask = (test_u == u)
        gt_items = set(map(int, test_i[mask].tolist()))

        # Seen slice for u (array of ints)
        seen_arr = get_seen_for_user(int(u), seen_u, seen_i, offset)
        
        # Tiny set just for this user (kept only for this iteration)
        seen_set = set(map(int, seen_arr.tolist()))

        # Build top list by skipping seen
        picked = []
        for it in pop_order:
            if it not in seen_set:
                picked.append(int(it))
                if len(picked) >= k_star:
                    break

        for K in Ks:
            topk = picked[:K]
            
            # Per-user metrics
            results[K]['rec']  += recall_at_k(topk, gt_items, K)
            results[K]['ndcg'] += ndcg_at_k(topk, gt_items, K)
            results[K]['hit']  += hitrate_at_k(topk, gt_items, K)
            
            # Catalog-level aggregates
            results[K]['cov_set'].update(topk)
            results[K]['nov'] += novelty_bits_for_list(topk, pop_counts)

    # Finalize
    out = {}
    for K in Ks:
        cov = len(results[K]['cov_set']) / float(n_items) if n_items > 0 else 0.0
        nov = results[K]['nov'] / float(n_users) if n_users > 0 else 0.0
        out[f'K={K}'] = {
            'users': n_users,
            'hit_rate': results[K]['hit'] / n_users if n_users else 0.0,
            'recall': results[K]['rec'] / n_users if n_users else 0.0,
            'ndcg': results[K]['ndcg'] / n_users if n_users else 0.0,
            'coverage': cov,
            'novelty_bits': nov,
        }

    outp = pathlib.Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f'Wrote popularity metrics → {outp}')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()