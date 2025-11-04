from __future__ import annotations
import argparse
import pathlib

import numpy as np
import pandas as pd

from movierec.models.lightgcn import fit_lightgcn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits_dir', default='artifacts/splits_default')
    ap.add_argument('--out_dir', default='artifacts/lightgcn')
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--n_layers', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=1)  # Change to 5-10
    ap.add_argument('--steps_per_epoch', type=int, default=1024)
    ap.add_argument('--batch_size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--l2', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')  # 'cpu', 'cuda'
    args = ap.parse_args()

    splits = pathlib.Path(args.splits_dir)
    train = pd.read_parquet(splits / 'train.parquet')
    n_users = int(train['u'].max()) + 1
    n_items = int(train['i'].max()) + 1

    print('Starting to train LightGCN')
    Ue, Ie = fit_lightgcn(
        train_df=train,
        n_users=n_users,
        n_items=n_items,
        dim=args.dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        device=args.device,
    )

    # Save LightGCN embeddings
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'user.npy', Ue)
    np.save(out / 'item.npy', Ie)
    print(f'Saved LightGCN embeddings to {out}')


if __name__ == '__main__':
    main()
