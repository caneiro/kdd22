# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

SEED = 42
HOME = Path.cwd()
print(HOME)
RAW_PATH = HOME / 'data/raw'
print(RAW_PATH)

def make_folds(df, n_splits=5):
    df['kfold'] = -1
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    return df

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    pub = pd.read_csv(RAW_PATH / 'public.csv')
    pup = make_folds(pub)
    pup.to_csv(RAW_PATH / 'public.csv', index=False)



if __name__ == '__main__':

    # not used in this stub but often useful for finding various files

    main()
