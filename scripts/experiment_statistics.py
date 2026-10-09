"""Audit saved experiments with paired, dependence-aware exploratory statistics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import t
from backtest import ROOT

KEYS=['territory_id','origin','date','horizon']
OUT=ROOT/'results/statistics'


def save_csv_atomic(frame,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temporary,index=False,encoding='utf-8-sig',chunksize=250)
    temporary.replace(path)


def holm(values):
    p=np.asarray(values,dtype=float);result=np.full(len(p),np.nan);valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid])]
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(order)-np.arange(len(order)))))
    return result


def sign_flip(sums):
    sums=np.asarray(sums,dtype=float);k=len(sums)
    if k<2:return np.nan
    observed=abs(sums.sum());tol=max(1e-10,observed*1e-12)
    if k<=18:
        signs=np.array(list(itertools.product([-1,1],repeat=k)),dtype=float)
        return float(np.mean(np.abs(signs@sums)>=observed-tol))
    rng=np.random.default_rng(20261009);signs=rng.choice([-1.,1.],(9999,k))
    return float((1+np.sum(np.abs(signs@sums)>=observed-tol))/10000)


def interval(sums,counts,length,repetitions,seed):
    k=len(sums)
    if k<max(3,2*length):return np.nan,np.nan
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,k-length+1,(repetitions,math.ceil(k/length)))
    indices=(starts[:,:,None]+np.arange(length)[None,None,:]).reshape(repetitions,-1)[:,:k]
    draws=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    return tuple(np.quantile(draws,[.025,.975]))


def normalize(frame):
    frame=frame.copy()
    for c in ['origin','date']:frame[c]=pd.to_datetime(frame[c])
    if frame.duplicated(KEYS).any():
        duplicates=frame[frame.duplicated(KEYS,keep=False)]
        for c in ['actual','prediction']:
            if c in frame and (duplicates.groupby(KEYS)[c].nunique()>1).any():
                raise ValueError('Conflicting duplicate forecasts: '+c)
        frame=frame.drop_duplicates(KEYS)
    return frame


def windows(g):
    masks={'public_origins_jun_sep':g.origin.between('2024-06-01','2024-09-01'),
            'later_origins':g.origin>pd.Timestamp('2024-09-01'),
            'targets_jun_sep':g.date.between('2024-06-01','2024-09-01'),
            'targets_oct_nov':g.date.between('2024-10-01','2024-11-01'),
            'targets_dec':g.date==pd.Timestamp('2024-12-01'),
            'targets_jan_may':g.date.between('2024-01-01','2024-05-01'),
            'all_available':np.ones(len(g),dtype=bool)}
    output={name:mask&g.horizon.isin([1,2,3]) for name,mask in masks.items()}
    for h in sorted(set(g.horizon)-{1,2,3}):
        output.update({name+f'/h{h}':mask&(g.horizon==h) for name,mask in masks.items()})
    return output

