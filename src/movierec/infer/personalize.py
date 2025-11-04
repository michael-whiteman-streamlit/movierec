from __future__ import annotations
import re

import numpy as np
import pandas as pd


NOW = 2025.0
YEAR_RE = re.compile(r'\s*\(\d{4}\)\s*$')


def apply_mmr(
    cand_ids: list[int],
    scores: np.ndarray,
    item_emb: np.ndarray,
    topk: int,
    lambda_div: float = 0.8,
):
    """
    Select top-k diverse yet relevant items using Maximal Marginal Relevance (MMR).
    Balances between relevance (scores) and diversity (dissimilarity).
    """
    # If topk is trivial or diversification negligible, return top-k by score
    if topk <= 0 or len(cand_ids) <= topk or lambda_div >= 0.999:
        return cand_ids[:topk]

    # Prepare candidate embeddings and state
    ids = [int(x) for x in cand_ids]
    V = item_emb[ids]  # Assume embeddings are L2-normalized
    picked = []  # Indices of selected items
    remaining = list(range(len(ids)))
    sims_cache = None  # Cosine similarity cache

    for _ in range(topk):
        if not picked:
            # First item is based purely on highest relevance
            best_idx = int(np.argmax(scores[remaining]))
            picked.append(remaining.pop(best_idx))
            continue

        # Lazily compute cosine similarity between all candidates
        if sims_cache is None:
            sims_cache = V @ V.T  # [n, n] cosine similarity matrix

        # Evaluate MMR for each remaining candidate
        best_val, best_pos = -1e9, -1
        for jpos, j in enumerate(remaining):
            max_sim = float(np.max(sims_cache[j, picked]))  # Most similar picked item
            mmr = lambda_div * float(scores[j]) - (1.0 - lambda_div) * max_sim
            if mmr > best_val:
                best_val, best_pos = mmr, jpos
        
        # Add the best candidate (highest MMR)
        picked.append(remaining.pop(best_pos))

    # Return top-k selected item IDs    
    return [ids[p] for p in picked]


def because_titles(cand_i: int, liked_ids: list[int], item_emb: np.ndarray, items_df: pd.DataFrame, k: int = 2) -> str:
    """Return a short explanation: the K liked titles nearest to this candidate."""
    if not liked_ids:
        return ""
    v = item_emb[cand_i]
    L = item_emb[liked_ids]
    sims = (L @ v)  # Cosine if normalized
    top = np.argsort(-sims)[:k]
    names = [re.sub(r"\s*\(\d{4}\)\s*$", "", str(items_df.loc[int(liked_ids[t]), "title"])) for t in top]
    return "Because you liked " + " & ".join(names)


def build_personal_vec(
    item_emb: np.ndarray,
    likes: list[int],
    dislikes: list[int],
    learned_user: np.ndarray | None = None,
    alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.2,
    recency_weights: list[float] | None = None,
) -> np.ndarray | None:
    """Construct a personalized embedding by combining liked, disliked, and learned user vectors with weights."""
    # If no user signals available, return None
    if not likes and not dislikes and learned_user is None:
        return None
    
    # Initialize user vector
    vec = np.zeros(item_emb.shape[1], dtype=np.float32)

    # Positive (liked) items
    if likes:
        L = item_emb[likes]
        if recency_weights is None:
            # Simple mean of liked item embeddings
            l_vec = L.mean(axis=0)
        else:
            # Weighted mean if recency weights provided
            w = np.asarray(recency_weights, dtype=np.float32)
            w = w[:len(likes)] / (w[:len(likes)].sum() + 1e-8)
            l_vec = (L[:len(w)] * w[:, None]).sum(axis=0)
        vec += alpha * l_vec  # Scale contribution by alpha

    # Negative (disliked) items
    if dislikes:
        D = item_emb[dislikes]
        d_vec = D.mean(axis=0)
        vec -= beta * d_vec  # Subtract weighted average of disliked items

    # Learned user embedding (from model)
    if learned_user is not None:
        vec += gamma * learned_user  # Blend learned representation

    # Normalize to unit length for cosine similiarity use
    vec = vec.astype(np.float32, copy=False)
    n = np.linalg.norm(vec) + 1e-12
    return vec / n


def explain_recommendation(
        cand_i: int,
        liked_ids: list[int],
        item_emb: np.ndarray,
        items_df: pd.DataFrame,
        k_nn: int = 2
    ) -> str:
    """
    Make a short, readable explanation:
      - the top-k liked titles nearest to this candidate
      - salient shared genres
      - rough year proximity
    """
    if not liked_ids:
        return ''
    # Determine nearest liked anchors
    anchors = _top_k_liked_neighbors(cand_i, liked_ids, item_emb, k=k_nn)
    anchor_titles = [_strip_year(items_df.loc[a, 'title']) for a in anchors]
    left = 'Because you liked ' + ' & '.join(anchor_titles) if anchor_titles else ''

    # Determine genre overlap
    g = _shared_genres(cand_i, anchors or liked_ids, items_df, top_n=2)
    mid = f'shares {", ".join(g)}' if g else ''

    # Calculate year proximity
    yr = _year_rationale(cand_i, anchors or liked_ids, items_df)
    right = yr or ''

    # Join non-empty parts with bullets
    parts = [p for p in (left, mid, right) if p]
    return ' • '.join(parts)


def rerank_candidates(
    user_vec: np.ndarray,
    cand_ids: list[int],
    item_emb: np.ndarray,
    base_scores: np.ndarray,
    items_df: pd.DataFrame,
    genre_prefs: dict[str, float] | None = None,
    year_pref: dict | None = None,
    watchlist_ids: set[int] | None = None,
    weights: dict = None,
    recency_half_life_years: float = 8.0
):
    """
    Build a blended score from:
      - base model score
      - cosine similarity to personalized user vec
      - genre alignment
      - alignment to user's preferred year (gaussian)
      - novelty (inverse popularity)
      - recency (newer release year -> higher score, exponential decay)

    Note: diversity is applied as a separate greedy step (MMR), but not inside this function.
    """    

    if weights is None:
        weights = {"base":1.0, "sim":0.7, "genre":0.2, "year":0.2, "novel":0.1, "weights": 0.5}

    cand_ids = [int(x) for x in cand_ids]
    
    # Align metadata to candidates
    idxer = items_df.index.get_indexer(cand_ids)  # -1 if missing
    mask = idxer >= 0
    if not np.any(mask):
        return cand_ids, np.asarray(base_scores, dtype=np.float32)

    cand_ids = [cid for cid, m in zip(cand_ids, mask) if m]
    idxer = idxer[mask]

    sims = (item_emb[cand_ids] @ user_vec).astype(np.float32)
    base_scores = np.asarray(base_scores, dtype=np.float32)[mask]
    meta = items_df.iloc[idxer]

    # Genre alignment
    gscore = np.zeros_like(sims)
    if genre_prefs:
        genres_col = meta["genres"].astype("string").fillna("")
        
        # Compute simple avg of weights across present genres
        def _gavg(s: str) -> float:
            if not s: return 0.0
            parts = [p.strip() for p in s.split("|") if p.strip()]
            if not parts: return 0.0
            return float(sum(genre_prefs.get(p, 0.0) for p in parts) / len(parts))
        gscore = np.array([_gavg(s) for s in genres_col.to_list()], dtype=np.float32)

    # Year preference
    yscore = np.zeros_like(sims)
    if year_pref:
        mu = float(year_pref.get("mu", 2000))
        sg = max(1.0, float(year_pref.get("sigma", 20)))
        years = meta["year"].astype("float32").fillna(mu).to_numpy()
        yscore = np.exp(-0.5 * ((years - mu) / sg) ** 2).astype(np.float32)

    # Novelty (inverse popularity)
    if "__pop__" in meta.columns:
        pop = meta["__pop__"].fillna(0).to_numpy(dtype=np.float32)
        nov = -np.log(np.maximum(1.0, pop))
    else:
        nov = np.zeros_like(sims)

    # Recency (we favor newer years with exponential decay by half-life)
    rec = np.zeros_like(sims)
    if 'year' in meta.columns and weights.get('recency', 0.0) > 0.0:
        cy = meta['year'].astype('float32').to_numpy()
        
        # If year is missing, give a neutral-ish 0.5
        age = np.maximum(0.0, NOW - np.nan_to_num(cy, nan=NOW))  # missing year -> age 0 -> score 1.0
        
        # Exponential decay: score = 0.5 ** (age / half_life)
        hl = max(0.1, float(recency_half_life_years))
        rec = np.power(0.5, age / hl).astype(np.float32)

    # Determine watchlist boost (a binary 0/1)
    wv = np.zeros_like(sims)
    if watchlist_ids:
        wl_set = set(int(x) for x in watchlist_ids)
        wv = np.array([1.0 if int(cid) in wl_set else 0.0 for cid in cand_ids], dtype=np.float32)

    final = (
        weights['base']  * base_scores +
        weights['sim']   * sims +
        weights['genre'] * gscore +
        weights['year']  * yscore +
        weights['novel'] * nov +
        weights.get('recency', 0.0) * rec +
        weights.get('watch', 0.0)   * wv
    )

    order = np.argsort(-final)
    ranked = [int(cand_ids[i]) for i in order]
    return ranked, final[order]


def _shared_genres(cand_i: int, ref_ids: list[int], items_df: pd.DataFrame, top_n: int = 2) -> list[str]:
    """Compute the most salient shared genres between candidate and a few reference liked items."""
    cand_g = set(_split_genres(items_df.loc[cand_i, 'genres']))
    if not cand_g or not ref_ids:
        return []
    
    # Weight genres by how frequently they appear among the refs
    counts = {}
    for rid in ref_ids:
        for g in set(_split_genres(items_df.loc[rid, 'genres'])):
            if g in cand_g:
                counts[g] = counts.get(g, 0) + 1
    if not counts:
        return []
    
    # Return top-N by frequency (ties arbitrary), stable order
    return [g for g, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]


def _split_genres(s: str) -> list[str]:
    """Split a MovieLens 'A|B|C' genre string into clean tokens."""
    if pd.isna(s) or not str(s).strip():
        return []
    return [p.strip() for p in str(s).split('|') if p.strip()]


def _strip_year(title: str) -> str:
    """Remove trailing (YYYY) from MovieLens titles for cleaner display."""
    return YEAR_RE.sub('', str(title or ''))


def _top_k_liked_neighbors(cand_i: int, liked_ids: list[int], item_emb: np.ndarray, k: int = 2) -> list[int]:
    """Return indices in liked_ids that are closest to cand_i by cosine (assuming item_emb are L2-normalized)."""
    if not liked_ids:
        return []
    v = item_emb[cand_i]
    L = item_emb[liked_ids]
    sims = L @ v  # cosine since L2-normalized
    order = np.argsort(-sims)[:k]
    return [int(liked_ids[idx]) for idx in order]


def _year_rationale(cand_i: int, ref_ids: list[int], items_df: pd.DataFrame) -> str | None:
    """Describe how close the candidate's year is to the nearest (or mean) of the reference liked years."""
    cy = items_df.loc[cand_i, 'year']
    if pd.isna(cy) or not ref_ids:
        return None
    ref_years = [items_df.loc[rid, 'year'] for rid in ref_ids if not pd.isna(items_df.loc[rid, 'year'])]
    if not ref_years:
        return None
    
    # Use the min distance to any ref year (more forgiving than mean)
    diff = min(abs(int(cy) - int(y)) for y in ref_years)
    if diff <= 2:
        return 'same era'
    if diff <= 5:
        return 'near in time'
    
    # Tag by decade if farther
    return f'close to the { (int(cy)//10)*10 }s'