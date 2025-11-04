from __future__ import annotations
import io
import os
import pathlib
from urllib.request import urlopen
import zipfile


ML32M_URL = 'https://files.grouplens.org/datasets/movielens/ml-32m.zip'
ML1M_URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
MLSMALL_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'


def download(url: str, out_dir: str | os.PathLike) -> pathlib.Path:
    # Ensure that target directory exists
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data from specified URL
    print(f"Downloading {url} ...")
    with urlopen(url) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(out_dir)
    return out_dir
