from __future__ import annotations
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn


class BPRDataset:
    """Samples (user, positive, negative) triplets for Bayesian Personalized Ranking training."""
    def __init__(self, train_df: pd.DataFrame, n_users: int, n_items: int, rng: np.random.Generator):
        self.n_users = n_users
        self.n_items = n_items
        self.rng = rng

        # Build positives set per user
        self.pos = defaultdict(set)
        for key, g in train_df.groupby(['u']):
            uid = key[0] if isinstance(key, tuple) else key   # <- handle tuple keys
            uid = int(uid)
            self.pos[uid].update(g['i'].tolist())

        # Only sample from users that have >= 1 positive
        self.users = np.array(sorted([u for u, s in self.pos.items() if len(s) > 0]), dtype=np.int64)
        if self.users.size == 0:
            raise ValueError('No users with positive interactions in train_df.')

    def sample_batch(self, batch_size: int):
        """Sample a batch of (user, positive item, negative item) triplets."""
        # Choose users by index into the compact users array
        u = self.users[self.rng.integers(0, len(self.users), size=batch_size)]

        i = np.empty(batch_size, dtype=np.int64)
        j = np.empty(batch_size, dtype=np.int64)

        for k, uu in enumerate(u):
            pos_list = list(self.pos[uu])
            i[k] = pos_list[self.rng.integers(0, len(pos_list))]

            # Draw a negative not in pos
            while True:
                cand = self.rng.integers(0, self.n_items)
                if cand not in self.pos[uu]:
                    j[k] = cand
                    break

        return torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(j)


class BPRMF(nn.Module):
    """Matrix factorization model for BPR with user/item embeddings and item bias."""
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, u, i):
        """Compute predicted preference scores for user-item pairs."""
        return (self.user(u) * self.item(i)).sum(dim=1) + self.item_bias(i).squeeze(-1)

    def bpr_loss(self, u, i, j, l2=1e-4):
        """Compute BPR loss with L2 regularization for (user, pos, neg) triplets."""
        s_pos = self.score(u, i)
        s_neg = self.score(u, j)
        loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()
        reg = l2 * (self.user(u).pow(2).mean() + self.item(i).pow(2).mean() + self.item(j).pow(2).mean())
        return loss + reg


def fit_bpr(train_df: pd.DataFrame, n_users: int, n_items: int,
            dim=64, epochs=10, steps_per_epoch=1024, batch_size=2048,
            lr=5e-3, l2=1e-4, seed=42, device='cpu'):
    """Train a BPRMF model using sampled triplets from BPRDataset."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    ds = BPRDataset(train_df, n_users, n_items, rng=rng)
    model = BPRMF(n_users, n_items, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs+1):
        losses = []
        for _ in range(steps_per_epoch):
            u, i, j = ds.sample_batch(batch_size)
            u, i, j = u.to(device), i.to(device), j.to(device)
            opt.zero_grad()
            loss = model.bpr_loss(u, i, j, l2=l2)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f'[BPR] epoch {ep:02d} loss={np.mean(losses):.4f}')
    return model
