#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path('')
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Cell 1: TARGET Data Dimensionality Reduction")
print("="*60)

mir_raw  = pd.read_csv(DATA_DIR / 'TARGET_miRNA_process.csv', index_col=0)
meth_raw = pd.read_csv(DATA_DIR / 'TARGET_meth_process.csv', index_col=0)
mrna_raw = pd.read_csv(DATA_DIR / 'TARGET_mRNA_process.csv', index_col=0)
cli      = pd.read_csv(DATA_DIR / 'TARGET_clinical_delete_process.csv', index_col=0)

common_idx = mir_raw.index.intersection(meth_raw.index).intersection(mrna_raw.index)
mir  = mir_raw.loc[common_idx]
meth = meth_raw.loc[common_idx]
mrna = mrna_raw.loc[common_idx]
cli  = cli.loc[common_idx]

status_map = {
    '1:DECEASED': 1, '0:LIVING': 0,
    'DECEASED': 1, 'LIVING': 0,
    'Dead': 1, 'Alive': 0,
    '1': 1, '0': 0
}
e = cli['os_status'].astype(str).map(status_map).values

print(f"Samples: {len(common_idx)}")
print(f"Original features: miRNA={mir.shape[1]}, meth={meth.shape[1]}, mRNA={mrna.shape[1]}")

def reduce_to_dim(df, y, target_dim, name):
    print(f"\n[{name}] Target dimension: {target_dim}")
    df = df.dropna(axis=1, how='all')
    X = SimpleImputer(strategy='median').fit_transform(df.values.astype(float))
    print(f"  After imputation: {X.shape[1]}")
    X = VarianceThreshold(threshold=0.05).fit_transform(X)
    print(f"  After variance threshold: {X.shape[1]}")
    if X.shape[1] > target_dim:
        if len(np.unique(y)) > 1 and np.sum(y) > 0 and np.sum(1-y) > 0:
            try:
                X = SelectKBest(f_classif, k=target_dim).fit_transform(X, y)
                print(f"  After ANOVA: {X.shape[1]}")
            except Exception as ex:
                print(f"  ANOVA failed ({ex}), truncating to {target_dim}")
                X = X[:, :target_dim]
        else:
            print(f"  No label variation, truncating to {target_dim}")
            X = X[:, :target_dim]
    elif X.shape[1] < target_dim:
        print(f"  Warning: {X.shape[1]} < {target_dim}, zero-padding")
        pad = np.zeros((X.shape[0], target_dim))
        pad[:, :X.shape[1]] = X
        X = pad
    X = MinMaxScaler().fit_transform(X)
    return X

X_mir  = reduce_to_dim(mir,  e, 200, 'miRNA')
X_meth = reduce_to_dim(meth, e, 600, 'Methylation')
X_mrna = reduce_to_dim(mrna, e, 600, 'mRNA')

pd.DataFrame(X_mir,  index=common_idx, columns=[f'f{i}' for i in range(200)]).to_csv(RESULT_DIR / 'TARGET_X_mir_200.csv')
pd.DataFrame(X_meth, index=common_idx, columns=[f'f{i}' for i in range(600)]).to_csv(RESULT_DIR / 'TARGET_X_meth_600.csv')
pd.DataFrame(X_mrna, index=common_idx, columns=[f'f{i}' for i in range(600)]).to_csv(RESULT_DIR / 'TARGET_X_mrna_600.csv')

np.save(RESULT_DIR / 'TARGET_X_mir_200.npy', X_mir)
np.save(RESULT_DIR / 'TARGET_X_meth_600.npy', X_meth)
np.save(RESULT_DIR / 'TARGET_X_mrna_600.npy', X_mrna)

pd.Series(common_idx, name='patient_id').to_csv(RESULT_DIR / 'TARGET_common_index.csv', index=False)

print(f"\nDone. Saved to: {RESULT_DIR}")