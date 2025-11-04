from __future__ import annotations
import pathlib as P
import pandas as pd
import numpy as np


RAW = P.Path('data/raw') / 'ml-1m'
PROC = P.Path('data/processed') / 'ml-1m'
ART = P.Path('artifacts') / 'ml-1m'
(PROC / 'mappings').mkdir(parents=True, exist_ok=True)
(ART / 'maps').mkdir(parents=True, exist_ok=True)
(ART / 'splits').mkdir(parents=True, exist_ok=True)


def _load_ratings_ml1m() -> pd.DataFrame:
    p = RAW / 'ratings.dat'
    # UserID::MovieID::Rating::Timestamp
    df = pd.read_csv(p, sep='::', engine='python',
                     names=['user','item','rating','timestamp'],
                     encoding='latin-1')
    return df

def _load_movies_ml1m() -> pd.DataFrame:
    p = RAW / 'movies.dat'
    # MovieID::Title (Year)::Genres
    df = pd.read_csv(p, sep='::', engine='python',
                     names=['item','title','genres'],
                     encoding='latin-1')
    return df

def _norm_genres(s: str) -> str:
    if pd.isna(s) or not str(s).strip():
        return ''
    parts = [p.strip() for p in str(s).split('|') if p.strip()]
    return ' | '.join(parts)

def main():
    # Load raw data
    ratings = _load_ratings_ml1m()
    movies  = _load_movies_ml1m()

    # Build contiguous id maps
    uid_map = pd.Series(
        data=np.arange(ratings['user'].nunique(), dtype=np.int64),
        index=pd.Index(sorted(ratings['user'].unique()), name='user'),
        name='u'
    )
    iid_map = pd.Series(
        data=np.arange(movies['item'].nunique(), dtype=np.int64),
        index=pd.Index(sorted(movies['item'].unique()), name='item'),
        name='i'
    )

    # Save maps in both processed and artifacts (parquet preferred; json for legacy)
    (PROC/'mappings').mkdir(parents=True, exist_ok=True)
    uid_map.to_json(PROC / 'mappings' / 'user_to_idx.json')
    iid_map.to_json(PROC / 'mappings' / 'item_to_idx.json')
    uid_map.to_frame().to_parquet(ART / 'maps' / 'uid_map.parquet')
    iid_map.to_frame().to_parquet(ART / 'maps' / 'iid_map.parquet')

    # Remap ratings to contiguous ids and save processed interactions
    df = ratings.merge(uid_map.reset_index(), on='user', how='left') \
                .merge(iid_map.reset_index(), on='item', how='left')
    inter = df[['u', 'i', 'rating', 'timestamp']].sort_values(['u', 'timestamp'])
    PROC.mkdir(parents=True, exist_ok=True)
    inter.to_parquet(PROC/'interactions.parquet', index=False)

    # Make temporal leave-last splits per user
    def _split_user(g: pd.DataFrame):
        # Last becomes test, second to last becomes val (if exists), rest becomes train
        if len(g) >= 2:
            val = g.iloc[[-2]]
            test = g.iloc[[-1]]
            train = g.iloc[:-2]
        else:
            val = g.iloc[[]]
            test = g.iloc[[-1]]
            train = g.iloc[[]]
        return train, val, test

    groups = inter.groupby('u', sort=False)
    trains, vals, tests = [], [], []
    for _, g in groups:
        tr, va, te = _split_user(g)
        trains.append(tr); vals.append(va); tests.append(te)

    train = pd.concat(trains, ignore_index=True)
    val   = pd.concat(vals, ignore_index=True)
    test  = pd.concat(tests, ignore_index=True)

    train[['u','i','timestamp']].to_parquet(ART/'splits'/'train.parquet', index=False)
    val[['u','i','timestamp']].to_parquet(ART/'splits'/'val.parquet', index=False)
    test[['u','i','timestamp']].to_parquet(ART/'splits'/'test.parquet', index=False)

    # Export items metadata aligned to contiguous item ids
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('Int64')
    movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    movies['genres'] = movies['genres'].apply(_norm_genres)

    items = movies.merge(iid_map.reset_index(), on='item', how='inner') \
                  .set_index('i').sort_index()[['title','genres','year']]

    # Attach simple popularity for novelty scoring
    pop = train['i'].value_counts()
    items['__pop__'] = pop.reindex(items.index).fillna(0).astype('int64')

    items.to_parquet(ART/'items.parquet')
    print('Wrote:')
    print(f'  {PROC / "interactions.parquet"}')
    print(f'  {ART / "maps/uid_map.parquet"}')
    print(f'  {ART / "maps/iid_map.parquet"}')
    print(f'  {ART / "splits/train.parquet"}, val.parquet, test.parquet')
    print(f'  {ART / "items.parquet"}')


if __name__ == '__main__':
    main()
