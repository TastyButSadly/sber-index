"""Compare runs on their intersection of forecast keys, never mismatched cohorts."""
import argparse
from pathlib import Path
import pandas as pd
from backtest import ROOT, summarize

parser = argparse.ArgumentParser()
parser.add_argument('runs', nargs='+', help='Directories containing predictions.parquet')
parser.add_argument('--output', default='results/comparison.csv')
args = parser.parse_args()
frames = [pd.read_parquet(ROOT/Path(p)/'predictions.parquet') for p in args.runs]
keys = ['territory_id', 'origin', 'date', 'horizon']
common = frames[0][keys].drop_duplicates()
for frame in frames[1:]:
    common = common.merge(frame[keys].drop_duplicates(), on=keys)
if common.empty:
    raise ValueError('Runs do not share any forecast keys')
joined = pd.concat([frame.merge(common, on=keys) for frame in frames], ignore_index=True)
if joined.duplicated(keys+['model']).any():
    raise ValueError('Duplicate model forecasts; pass each run once')
result = summarize(joined)
out = ROOT/args.output
out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out, index=False)
print(result.to_string(index=False))
