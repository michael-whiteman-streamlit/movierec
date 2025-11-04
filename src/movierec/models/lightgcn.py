from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn


class _UniformPairSampler:
    """Samples (user, positive, negative) triplets with uniform negative selection for BPR/LightGCN."""

    def __init__(self, train_df: pd.DataFrame, n_items: int, rng: np.random.Generator):
        self.rng = rng
        self.n_items = int(n_items)
        pos = {}
        for u, grp in train_df.groupby('u'):
            pos[int(u)] = set(map(int, grp['i'].values))
        self.users = np.array(sorted([u for u, s in pos.items() if s]), dtype=np.int64)
        self.pos = pos

    def sample(self, batch_size: int):
        """Sample a batch of (user, positive item, negative item) triplets."""
        u = self.users[self.rng.integers(0, len(self.users), size=batch_size)]
        i = np.empty(batch_size, dtype=np.int64)
        j = np.empty(batch_size, dtype=np.int64)
        for k, uu in enumerate(u):
            pos_list = list(self.pos[uu])
            i[k] = pos_list[self.rng.integers(0, len(pos_list))]
            
            # Uniform negative sampling
            while True:
                jj = self.rng.integers(0, self.n_items)
                if jj not in self.pos[uu]:
                    j[k] = jj
                    break
        return torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(j)


class LightGCN(nn.Module):
    """Light Graph Convolutional Network for collaborative filtering."""

    def __init__(self, n_users, n_items, dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.01)

    def propagate(self, A_hat):
        """Propagate embeddings through graph layers using normalized adjacency A_hat."""
        E0 = self.emb.weight
        layers = [E0]
        E = E0
        for _ in range(self.n_layers):
            E = torch.sparse.mm(A_hat, E)
            layers.append(E)

        # Calculate mean of layer embeddings
        E_final = torch.stack(layers, dim=0).mean(dim=0)
        return E_final

    def score(self, U, I):
        """Compute predicted scores for given user–item pairs using propagated embeddings."""
        # U: [B], I: [B]
        E = self.propagated  # Set during training loop for speed
        Ue, Ie = E[:self.n_users], E[self.n_users:]
        return (Ue[U] * Ie[I]).sum(dim=1)

    def split_user_item(self, E):
        """Split concatenated node embeddings into user and item embeddings."""
        return E[:self.n_users], E[self.n_users:]


def bpr_loss(u_emb: torch.Tensor, i_emb: torch.Tensor, j_emb: torch.Tensor, l2: float = 1e-4):
    """Compute BPR loss with L2 regularization for user–positive–negative triplets."""
    s_pos = (u_emb * i_emb).sum(dim=1)
    s_neg = (u_emb * j_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()
    reg = l2 * (u_emb.pow(2).mean() + i_emb.pow(2).mean() + j_emb.pow(2).mean())
    return loss + reg


def build_normalized_adj(n_users: int, n_items: int, train_df) -> torch.Tensor:
    """
    Build A_hat = D^{-1/2} A D^{-1/2} for the bipartite graph in sparse form.
    Returns a torch.sparse_coo_tensor of shape (n_users+n_items, n_users+n_items).
    """
    # User–item incidence (sparse)
    rows = train_df['u'].to_numpy(dtype=np.int32)
    cols = train_df['i'].to_numpy(dtype=np.int32)
    data = np.ones(len(train_df), dtype=np.float32)
    A_ui = sp.coo_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    # Bipartite adjacency:
    # [ 0  R ]
    # [ R^T 0 ]
    A = sp.bmat([[None, A_ui], [A_ui.T, None]], format='coo', dtype=np.float32)

    # Degree vector and symmetric normalization
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    Dmh = sp.diags(d_inv_sqrt.astype(np.float32))

    A_hat = (Dmh @ A @ Dmh).tocoo().astype(np.float32)

    # Convert to torch sparse
    indices = np.vstack([A_hat.row, A_hat.col]).astype(np.int64)
    values = A_hat.data.astype(np.float32)
    i = torch.from_numpy(indices)
    v = torch.from_numpy(values)
    shape = (n_users + n_items, n_users + n_items)
    A_hat_torch = torch.sparse_coo_tensor(i, v, size=shape).coalesce()
    return A_hat_torch


def fit_lightgcn(train_df: pd.DataFrame,
                 n_users: int,
                 n_items: int,
                 dim: int = 64,
                 n_layers: int = 3,
                 epochs: int = 15,
                 steps_per_epoch: int = 2048,
                 batch_size: int = 4096,
                 lr: float = 5e-3,
                 l2: float = 1e-4,
                 seed: int = 42,
                 device: str = 'cpu'):
    """
    Train LightGCN with BPR on propagated embeddings.
    Returns numpy arrays (U, I) suitable for your app.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device)

    # Graph (sparse) and embeddings (users+items as one table)
    A_hat = build_normalized_adj(n_users, n_items, train_df).to(device)
    num_nodes = int(n_users + n_items)
    emb = torch.nn.Embedding(num_nodes, dim, device=device)
    torch.nn.init.xavier_uniform_(emb.weight)

    opt = torch.optim.Adam(emb.parameters(), lr=lr)
    sampler = _UniformPairSampler(train_df, n_items=n_items, rng=rng)

    emb.train()
    for ep in range(1, epochs + 1):
        running = 0.0
        for step in range(steps_per_epoch):
            # Recompute LightGCN-propagated node embeddings each step (most correct, slowest)
            node_emb = lightgcn_propagate(emb, A_hat, n_layers)

            u, i, j = sampler.sample(batch_size)
            u = u.to(device, non_blocking=True)
            i = i.to(device, non_blocking=True)
            j = j.to(device, non_blocking=True)

            # Map item ids into the node index space [n_users .. n_users+n_items)
            iu = u
            ii = n_users + i
            ij = n_users + j

            u_emb = node_emb[iu]
            i_emb = node_emb[ii]
            j_emb = node_emb[ij]

            loss = bpr_loss(u_emb, i_emb, j_emb, l2=l2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu().item())
        print(f'[LightGCN] epoch {ep:02d} loss={running/steps_per_epoch:.4f}')

    # Final propagation for export
    with torch.no_grad():
        node_final = lightgcn_propagate(emb, A_hat, n_layers)
        Ue = node_final[:n_users].detach().cpu().numpy().astype(np.float32)
        Ie = node_final[n_users:].detach().cpu().numpy().astype(np.float32)
    return Ue, Ie


def lightgcn_propagate(emb_table: torch.nn.Embedding,
                       A_hat: torch.Tensor,
                       n_layers: int) -> torch.Tensor:
    """Propagate embeddings through n LightGCN layers using normalized adjacency A_hat."""
    x = emb_table.weight
    out = x
    cur = x
    for _ in range(n_layers):
        # Sparse × dense
        cur = torch.sparse.mm(A_hat, cur)
        out = out + cur
    out = out / (n_layers + 1)
    return out


def sample_triplets(train_df: pd.DataFrame, n_users:int, n_items:int, batch_size:int, rng):
    """Sample (user, positive, negative) triplets for BPR training from interaction data."""
    pos_dict = {}
    for u, g in train_df.groupby('u'):
        pos_dict[int(u)] = g['i'].to_numpy()

    users = rng.integers(0, n_users, size=batch_size)
    i = np.empty(batch_size, dtype=np.int64)
    j = np.empty(batch_size, dtype=np.int64)

    for k,u in enumerate(users):
        pos = pos_dict[u]
        i[k] = pos[rng.integers(0, len(pos))]

        while True:
            cand = rng.integers(0, n_items)
            if cand not in pos:
                j[k] = cand
                break
    return torch.from_numpy(users), torch.from_numpy(i), torch.from_numpy(j)


def split_user_item(node_emb: torch.Tensor, n_users: int):
    """Split concatenated node embeddings into user and item embeddings."""
    Ue = node_emb[:n_users]
    Ie = node_emb[n_users:]
    return Ue, Ie
