from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd


# Specify and make directories
RAW = pathlib.Path('data/raw')
PROC = pathlib.Path('data/processed') / 'ml-32m'
ART = pathlib.Path('artifacts') / 'ml-32m'
(ART / 'splits').mkdir(parents=True, exist_ok=True)
(ART / 'maps').mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
(PROC / 'mappings').mkdir(parents=True, exist_ok=True)

# Minimum interactions to keep a user
MIN_USER_INTERACTIONS = 5


def load_raw():
    """Load raw MovieLens ratings and movies CSVs from available dataset versions."""
    movies_p = _first_existing(
        RAW/'ml-32m'/'movies.csv',
        RAW/'ml-25m'/'movies.csv',
        RAW/'ml-20m'/'movies.csv',
        RAW/'ml-latest'/'movies.csv',
    )
    ratings_p = _first_existing(
        RAW/'ml-32m'/'ratings.csv',
        RAW/'ml-25m'/'ratings.csv',
        RAW/'ml-20m'/'ratings.csv',
        RAW/'ml-latest'/'ratings.csv',
    )
    if not movies_p or not ratings_p:
        raise FileNotFoundError('Put large MovieLens csvs under data/raw/(ml-32m|ml-25m|ml-20m|ml-latest)')
    
    # Read movie and rating data
    movies = pd.read_csv(movies_p)  # columns: movieId, title, genres
    ratings = pd.read_csv(ratings_p)  # columns: userId, movieId, rating, timestamp
    return ratings, movies


def build_maps(ratings: pd.DataFrame):
    """Create and save user/item ID to index mappings for later use."""
    # Build sorted index-based mappings
    uid = pd.Index(ratings['userId'].unique()).sort_values()
    iid = pd.Index(ratings['movieId'].unique()).sort_values()
    uid_map = pd.Series(range(len(uid)), index=uid, name='u')
    iid_map = pd.Series(range(len(iid)), index=iid, name='i')

    # Save to Parquet maps for efficient reuse
    uid_map.to_frame().to_parquet(ART / 'maps' / 'uid_map.parquet')
    iid_map.to_frame().to_parquet(ART / 'maps' / 'iid_map.parquet')    
    
    # Save to JSON for convenient script access
    uid_map.to_json(PROC/'mappings'/'user_to_idx.json')
    iid_map.to_json(PROC/'mappings'/'item_to_idx.json')
    return uid_map, iid_map


def extract_year(title: str | float) -> float | None:
    """Extract release year from a movie title string like 'Toy Story (1995)'."""
    if not isinstance(title, str):
        return np.nan
    m = pd.Series([title]).str.extract(r'\((\d{4})\)').iloc[0, 0]
    return float(m) if pd.notna(m) else np.nan


def make_splits(inter: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split user–item interactions into train, val, and test by recency per user."""
    # Add per-user rank by timestamp    
    inter = inter.copy()
    inter['rn'] = inter.groupby('u').cumcount() + 1
    last_rn = inter.groupby('u')['rn'].transform('max')

    # Most recent becomes test, second-most becomes val, rest becomes train
    test = inter[inter['rn'].eq(last_rn)]
    val  = inter[inter['rn'].eq(last_rn - 1)]
    keep = ~(inter.index.isin(test.index) | inter.index.isin(val.index))
    train = inter.loc[keep, ['u', 'i', 'timestamp']]
    return train, val[['u', 'i', 'timestamp']], test[['u', 'i', 'timestamp']]


def norm_genres(s) -> str:
    """Normalize and clean genre strings by stripping and joining with ' | '."""
    if pd.isna(s) or not str(s).strip():
        return ''
    parts = [p.strip() for p in str(s).split('|') if p.strip()]
    return ' | '.join(parts)


def _first_existing(*parts: pathlib.Path):
    """Iterate over paths to determine first which exists."""
    for p in parts:
        if p.exists():
            return p
    return None


def main():
    # Load raw data
    ratings, movies = load_raw()

    # Build contiguous id maps and write them
    uid_map, iid_map = build_maps(ratings)

    # Remap ratings to processed interactions
    inter = (
        ratings.join(uid_map, on='userId')
               .join(iid_map, on='movieId')
               [['u', 'i', 'rating', 'timestamp']]
               .astype({'u': 'int32', 'i': 'int32'})
               .sort_values(['u', 'timestamp'])
               .reset_index(drop=True)
    )

    # Filter very cold users
    if MIN_USER_INTERACTIONS and MIN_USER_INTERACTIONS > 1:
        keep_u = inter.groupby('u').size().loc[lambda s: s >= MIN_USER_INTERACTIONS].index
        inter = inter[inter['u'].isin(keep_u)].reset_index(drop=True)

    # Write a processed copy for parity/debugging
    inter.to_parquet(PROC / 'interactions.parquet', index=False)

    # Leave-one-out per user for val/test (latest 2), rest to train
    train, val, test = make_splits(inter)
    train.to_parquet(ART/'splits'/'train.parquet', index=False)
    val.to_parquet(ART/'splits'/'val.parquet', index=False)
    test.to_parquet(ART/'splits'/'test.parquet', index=False)

    # Align items metadata to contiguous item ids
    items = (
        movies.join(iid_map, on='movieId')
              .dropna(subset=['i'])
              .set_index('i')
              .sort_index()
              .copy()
    )
    items['year']  = items['title'].map(extract_year).round().astype('Int64')
    items['title'] = items['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    items['genres'] = items['genres'].fillna('').map(norm_genres)

    # Attach popularity for novelty scoring (using train to avoid leakage)
    pop = train['i'].value_counts()
    items['__pop__'] = pop.reindex(items.index).fillna(0).astype('int64')

    items[['title', 'genres', 'year', '__pop__']].to_parquet(ART/'items.parquet')

    # Report
    print('wrote:')
    print(f'  {PROC / "interactions.parquet"}')
    print(f'  {ART / "maps/uid_map.parquet"}')
    print(f'  {ART / "maps/iid_map.parquet"}')
    print(f'  {ART / "splits/train.parquet"}, val.parquet, test.parquet')
    print(f'  {ART / "items.parquet"}')


if __name__ == '__main__':
    main()
