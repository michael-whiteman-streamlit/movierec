import pandas as pd
import pathlib


def _load_ml1m(root: pathlib.Path) -> pd.DataFrame:
    """
    Load MovieLens-1M benchmark dataset.
    
    All ratings are contained in the file "ratings.dat" and are in the following format:
        
    UserID::MovieID::Rating::Timestamp

    - UserIDs range between 1 and 6040 
    - MovieIDs range between 1 and 3952
    - Ratings are made on a 5-star scale (whole-star ratings only)
    - Timestamp is represented in seconds since the epoch as returned by time(2)
    - Each user has at least 20 ratings
    """

    p = next((root / 'ml-1m').glob('ratings.dat'))
    df = pd.read_csv(p, sep='::', engine='python', names=['user', 'item', 'rating', 'ts'])
    return df


def _load_ml32m(root: pathlib.Path) -> pd.DataFrame:
    """
    Load MovieLens-32M benchmark dataset.
    
    All ratings are contained in the file "ratings.dat" and are in the following format:
        
    UserID::MovieID::Rating::Timestamp

    - UserIDs range between 1 and 6040 
    - MovieIDs range between 1 and 3952
    - Ratings are made on a 5-star scale (whole-star ratings only)
    - Timestamp is represented in seconds since the epoch as returned by time(2)
    - Each user has at least 20 ratings
    """

    p = next((root / 'ml-32m').glob('ratings.dat'))
    df = pd.read_csv(p, sep='::', engine='python', names=['user', 'item', 'rating', 'ts'])
    return df



def _load_mllatest(root: pathlib.Path) -> pd.DataFrame:
    """
    Load developmental MovieLens dataset (ml-latest-small).

    All ratings are contained in the file "ratings.csv".
    """
    p = next((root/'ml-latest-small').glob('ratings.csv'))
    df = pd.read_csv(p).rename(
        columns={
            'userId': 'user',
            'movieId': 'item',
            'timestamp': 'ts',
            }
        )
    return df


def load_ratings(raw_dir: str|pathlib.Path) -> pd.DataFrame:
    """
    Load desired ratings (MovieLens-1M benchmark or MovieLens-latest-small
    developmental dataset.)
    """
    root = pathlib.Path(raw_dir)
    if (root / 'ml-1m').exists():
        return _load_ml1m(root)
    elif (root / 'ml-latest-small').exists():
        return _load_mllatest(root)
    raise FileNotFoundError('Place MovieLens under data/raw/ (ml-1m or ml-latest-small)')


def reindex_ids(df: pd.DataFrame):
    """
    Map raw user/item ids to contiguous indices [0 ... n-1] and return reindexed data.
    Returns (df[['u','i','ts']], user_to_idx, item_to_idx).
    """
    # Map user/item ids to contiguous [0 ... n-1]
    uids = {u:i for i, u in enumerate(df['user'].unique())}
    iids = {it:i for i, it in enumerate(df['item'].unique())}
    df2 = df.copy()
    df2['u'] = df2['user'].map(uids)
    df2['i'] = df2['item'].map(iids)
    return df2[['u', 'i', 'ts']], uids, iids

def to_implicit(df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """
    Convert explicit ratings to implicit positives by thresholding (rating >= threshold).
    Returns a [user, item, ts] DataFrame of positive interactions sorted by timestamp.
    """
    df = df.copy()
    df['y'] = (df['rating'] >= threshold).astype(int)
    df = df[df['y'] == 1][['user', 'item', 'ts']].sort_values('ts').reset_index(drop=True)
    return df
