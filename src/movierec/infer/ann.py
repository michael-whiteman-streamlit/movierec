from __future__ import annotations

import faiss
import numpy as np


def build_ip_index(item_emb: np.ndarray, use_ivf: bool | None = None):
    """Build a FAISS inner-product index (flat or IVF) from item embeddings."""
    V = ensure_normalized_item_emb(item_emb)
    d = V.shape[1]
    use_ivf = V.shape[0] > 200_000 if use_ivf is None else use_ivf
    if use_ivf:
        nlist = max(64, int(np.sqrt(V.shape[0]) // 2))
        quant = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(V); idx.add(V)
        idx.nprobe = max(8, nlist // 20)
        return ('ivf', idx)
    else:
        idx = faiss.IndexFlatIP(d); idx.add(V)
        return ('flat_faiss', idx)


def ensure_normalized_item_emb(I: np.ndarray) -> np.ndarray:
    """Normalize item embeddings to unit length along feature dimensions."""
    I = I.astype('float32', copy=False)
    I = I / (np.linalg.norm(I, axis=1, keepdims=True) + 1e-12)
    return I


def query_index(index_obj, qvec: np.ndarray, topk: int = 100, forbid: set[int] | None = None):
    """Query FAISS index or dense matrix for top-k similar items excluding forbidden ones."""
    kind, idx = index_obj
    forbid = forbid or set()
    q = qvec.astype('float32', copy=False)
    if kind == 'flat_np':
        sims = idx @ q
        order = np.argsort(-sims)
        out = []
        for j in order:
            if int(j) in forbid: continue
            out.append(int(j))
            if len(out) >= topk: break
        return out
    D, I = idx.search(q[None, :], topk + len(forbid) + 2)
    cand = [int(i) for i in I[0] if i >= 0 and int(i) not in forbid]
    return cand[:topk]
