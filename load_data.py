import os

import numpy as np
import pandas as pd

DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'human_mrtt.csv')


def stan_data(cond):
    df = pd.read_csv(DATA_CSV)
    df = df[df['cond'] == cond]
    pids = df['pid'].unique()
    T = df.groupby('pid').size().iloc[0]
    inv = df.pivot(index='pid', columns='t', values='investment').loc[pids].to_numpy()
    k = df.pivot(index='pid', columns='t', values='k').loc[pids].to_numpy()
    return {'N': len(pids), 'T': int(T), 'inv': inv.astype(int), 'k': k.astype(int)}, pids
