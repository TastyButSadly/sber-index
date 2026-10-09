from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def load_panel(limit=0):
    df = pd.read_parquet(ROOT / 'data/processed/panel.parquet')
    cats = [c for c in df.columns if c not in ['territory_id', 'date']]
    dates = pd.DatetimeIndex(sorted(df.date.unique()))
    ids = sorted(df.territory_id.unique())
    if limit:
        ids = ids[:limit]
    index = pd.MultiIndex.from_product([ids, dates], names=['territory_id', 'date'])
    values = df.set_index(['territory_id', 'date']).reindex(index)[cats].to_numpy().reshape(len(ids), len(dates), len(cats))
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Panel has missing/nonpositive values; choose and document a causal missing-data policy first')
    return np.asarray(ids), dates, cats, values
