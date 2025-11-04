from __future__ import annotations
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from movierec.eval.metrics import (
    coverage_at_k,
    hitrate_at_k,
    ndcg_at_k,
    novelty_at_k,
    recall_at_k,
)


def load_seen(train: pd.DataFrame, val: pd.DataFrame):
    """Build a dict mapping each user to the set of items they've seen across train and val splits."""
    seen = {}
    for df in (train, val):
        for u, g in df.groupby('u'):
            s = seen.setdefault(int(u), set())
            s.update(g['i'].tolist())
    return seen


def topk_for_user(u, user_vec, item_mat, item_bias, seen_set, Kmax):
    """Return top-K unobserved item indices for user u based on dot-product scores and optional bias."""
    # Score = dot(u, V) + b (optional)
    scores = item_mat @ user_vec
    if item_bias is not None:
        scores = scores + item_bias
    
    # Filter seen (TRAIN + VAL)
    if seen_set:
        scores[list(seen_set)] = -1e30
    
    # Argpartition and then argsort within K window
    idx = np.argpartition(-scores, Kmax)[:Kmax]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_dir', default='artifacts/splits_default')
    parser.add_argument('--artifacts', default='artifacts/bpr_mf')
    parser.add_argument('--out_json', default='artifacts/metrics/bpr_mf.json')
    args = parser.parse_args()

    artifacts = pathlib.Path(args.artifacts)
    user = np.load(artifacts / 'user.npy')
    item = np.load(artifacts / 'item.npy')
    item_bias = np.load(artifacts / 'item_bias.npy') if (artifacts / 'item_bias.npy').exists() else None

    splits = pathlib.Path(args.splits_dir)
    train = pd.read_parquet(splits / 'train.parquet')
    val = pd.read_parquet(splits / 'val.parquet')
    test = pd.read_parquet(splits / 'test.parquet')

    seen = load_seen(train, val)
    gt = {int(u): set(g['i'].tolist()) for u, g in test.groupby('u')}
    n_items = int(max(train['i'].max(), val['i'].max(), test['i'].max())) + 1

    Ks = [10, 20]
    Kmax = max(Ks)

    per_user_topk = {}
    for u in gt.keys():
        per_user_topk[u] = topk_for_user(
            u=u,
            user_vec=user[u],
            item_mat=item,
            item_bias=item_bias,
            seen_set=seen.get(u, set()),
            Kmax=Kmax,
        )

    results={}
    for K in Ks:
        recalls, ndcgs, hitrates = [], [], []
        for u, topk in per_user_topk.items():
            gset = gt[u]
            recalls.append(recall_at_k(topk, gset, K))
            ndcgs.append(ndcg_at_k(topk, gset, K))
            hitrates.append(hitrate_at_k(topk, gset, K))
        coverage = coverage_at_k([v[:K] for v in per_user_topk.values()], n_items)
        
        # Compute novelty using train popularity
        pop = train["i"].value_counts().to_dict()
        novelty = novelty_at_k([v[:K] for v in per_user_topk.values()], pop, K)
        results[f'Recall@{K}']=float(np.mean(recalls))
        results[f'NDCG@{K}']=float(np.mean(ndcgs))
        results[f'HitRate@{K}']=float(np.mean(hitrates))
        results[f'Coverage@{K}']=float(coverage)
        results[f'Novelty@{K}']=float(novelty)

    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
