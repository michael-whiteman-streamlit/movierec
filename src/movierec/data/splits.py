from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int


def temporal_leave_last(df_ui_ts: pd.DataFrame, n_val=1, n_test=1) -> Splits:
    """
    Per-user temporal split: sort by (u, ts) and take the last n_test as test,
    the preceding n_val as validation, and the rest as train. Users with fewer than
    n_val + n_test interactions are kept entirely in train. Returns Splits(train, val,
    test, n_users, n_items).
    """
    # Input dataframe has columns u, i, ts
    df = df_ui_ts.sort_values(['u', 'ts']).reset_index(drop=True)
    groups = df.groupby('u')
    train_rows, val_rows, test_rows = [], [], []
    for u, g in groups:
        if len(g) < (n_val + n_test):
            # Too short, so put all into train
            train_rows.append(g)
            continue
        train_rows.append(g.iloc[:-(n_test + n_val)])
        val_rows.append(g.iloc[-(n_test + n_val): -n_test])
        test_rows.append(g.iloc[-n_test:])

    train = pd.concat(train_rows).reset_index(drop=True) if train_rows else df.iloc[0:0]
    val = pd.concat(val_rows).reset_index(drop=True) if val_rows else df.iloc[0:0]
    test = pd.concat(test_rows).reset_index(drop=True) if test_rows else df.iloc[0:0]
    n_users = int(df['u'].max()) + 1 if not df.empty else 0
    n_items = int(df['i'].max()) + 1 if not df.empty else 0
    return Splits(train, val, test, n_users, n_items)
