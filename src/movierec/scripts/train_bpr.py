from __future__ import annotations
import argparse
import pathlib

import numpy as np
import pandas as pd

from movierec.models.bpr_mf import fit_bpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits_dir', default='artifacts/splits_default')
    ap.add_argument('--out_dir', default='artifacts/bpr_mf')
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--steps_per_epoch', type=int, default=1024)
    ap.add_argument('--batch_size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--l2', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    splits = pathlib.Path(args.splits_dir)
    train = pd.read_parquet(splits / 'train.parquet')
    n_users = int(train['u'].max()) + 1
    n_items = int(train['i'].max()) + 1

    print('Starting to train Bayesian Personalized Ranking – Matrix Factorization (BPR-MF)')
    model = fit_bpr(
        train_df=train,
        n_users=n_users,
        n_items=n_items,
        dim=args.dim,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        device=args.device,
    )

    # Save embeddings
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'user.npy', model.user.weight.detach().cpu().numpy())
    np.save(out / 'item.npy', model.item.weight.detach().cpu().numpy())
    np.save(out / 'item_bias.npy', model.item_bias.weight.detach().cpu().numpy().squeeze(-1))
    print(f'Saved embeddings to {out}')


if __name__ == '__main__':
    main()
