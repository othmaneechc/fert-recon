import numpy as np
import pandas as pd

def temporal_split(df, train_max_year, val_year, test_year_min, test_year_max=None, train_min_year=None):
    mask_train = df.year <= train_max_year
    if train_min_year is not None:
        mask_train &= df.year >= train_min_year
    tr = df[mask_train]
    va = df[df.year == val_year]
    mask_test = df.year >= test_year_min
    if test_year_max is not None:
        mask_test &= df.year <= test_year_max
    te = df[mask_test]
    return tr, va, te


def temporal_leaveout(df, val_year, test_year):
    mask_val = df.year == val_year
    mask_test = df.year == test_year
    va = df[mask_val]
    te = df[mask_test]
    tr = df[~(mask_val | mask_test)]
    return tr, va, te

def spatial_kfold(df, kfolds=5, fold_index=0):
    # deterministic spatial hash split by pixel_id
    pid = df["pixel_id"].values
    fold = (pid * 2654435761) % kfolds
    tr = df[fold != fold_index]
    te = df[fold == fold_index]
    # make a validation split from train by year median
    val_year = int(np.median(tr["year"].unique()))
    va = tr[tr.year == val_year]
    tr = tr[tr.year != val_year]
    return tr, va, te
