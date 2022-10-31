# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path


def main(train_pixels_file, test_pixels_file):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # Project Directory
    project_dir = Path(__file__).resolve().parents[2]
    raw_dir = project_dir / '/data/raw'

    # Train Pixels
    train_pixels = pd.read_csv(raw_dir / train_pixels_file)
    print(train_pixels.shape)
    print(train_pixels.head())

    # Test Pixels
    train_pixels = pd.read_csv(raw_dir / train_pixels_file)
    print(train_pixels.shape)
    print(train_pixels.head())



from sklearn.model_selection import KFold

def make_folds(df, n_splits=5):
    df['kfold'] = -1
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    return df

pub = make_folds(pub)

pub.head()
    
    



if __name__ == '__main__':

    # not used in this stub but often useful for finding various files

    main()
