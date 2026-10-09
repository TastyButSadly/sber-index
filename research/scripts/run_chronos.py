"""Multivariate Chronos-2 rolling backtest, using the same panel and dates as baselines."""
from pathlib import Path
import os
os.environ.setdefault('ONEDNN_PRIMITIVE_CACHE_CAPACITY','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
import json
import numpy as np
import pandas as pd
import torch
from backtest import ROOT, load_panel, summarize
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=8, help='0 = all territories; default is a CPU smoke run')
    parser.add_argument('--origins', nargs='+', default=['2024-05-01'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='results/chronos_smoke')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--mode', choices=['multivariate','univariate','cross','residual'], default='multivariate')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.mkldnn.enabled = False
    from chronos import Chronos2Pipeline
    cfg = json.loads((ROOT / 'configs/backtest.json').read_text(encoding='utf-8'))
    ids, dates, cats, values = load_panel(args.limit)
    target = cats.index(cfg['target'])
    pipeline = Chronos2Pipeline.from_pretrained(str(ROOT / 'models/chronos-2-small'), device_map=args.device)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    records = []
    completed = set()
    if args.resume and (out/'predictions.parquet').exists():
        existing = pd.read_parquet(out/'predictions.parquet')
        if set(existing.model) != {'chronos2_small_'+args.mode} or set(existing.territory_id) != set(ids):
            raise ValueError('Resume checkpoint has a different mode or municipality cohort')
        records.append(existing)
        completed = set(pd.to_datetime(existing.origin).dt.strftime('%Y-%m-%d'))
    started = time.time()
    for origin_date in args.origins:
        if origin_date in completed:
            print(f'Resumed existing origin {origin_date}',flush=True)
            continue
        origin = dates.get_loc(pd.Timestamp(origin_date))
        if origin < 11:
            raise ValueError('Need 12 observed months to compare with seasonal naive')
        hs = [h for h in cfg['horizons'] if origin+h < len(dates)]
        if not hs:
            raise ValueError('No observed targets for this origin')
        context = pd.DataFrame(values[:, :origin+1].reshape(-1, len(cats)), columns=cats)
        context['territory_id'] = np.repeat(ids, origin+1)
        context['date'] = np.tile(dates[:origin+1], len(ids))
        targets = [cfg['target']] if args.mode == 'univariate' else cats
        if args.mode == 'univariate':
            context = context[['territory_id','date',cfg['target']]]
        if args.mode == 'residual':
            logged = np.log(values[:, :origin+1, :])
            factor = np.median(logged, axis=0)
            context[cats] = (logged-factor[None,:,:]).reshape(-1,len(cats))
        forecast = pipeline.predict_df(context, prediction_length=max(hs),
            quantile_levels=[0.1, 0.5, 0.9], id_column='territory_id', timestamp_column='date',
            target=targets, batch_size=args.batch_size, context_length=origin+1, cross_learning=args.mode=='cross')
        forecast.to_parquet(out / f'raw_{origin_date}.parquet', index=False)
        forecast = forecast[forecast['target_name'] == cfg['target']]
        for h in hs:
            selected = forecast[forecast.date == dates[origin+h]].set_index('territory_id').reindex(ids)
            prediction = selected['0.5'].to_numpy()
            if args.mode == 'residual':
                from forecast_common import factor_predict
                selection = json.loads((ROOT/'results/hypotheses/selection.json').read_text(encoding='utf-8'))
                common_future = factor_predict(factor[:,target],origin,h,selection['factor'])
                prediction = np.exp(prediction+common_future)
            if not np.isfinite(prediction).all():
                raise ValueError('Missing/nonfinite Chronos predictions')
            records.append(pd.DataFrame({'territory_id': ids, 'origin': dates[origin], 'date': dates[origin+h],
                'horizon': h, 'model': 'chronos2_small_'+args.mode, 'prediction': prediction,
                'actual': values[:, origin+h, target], 'seasonal_value': values[:, origin+h-12, target],
                'split': 'development' if dates[origin+h] <= pd.Timestamp(cfg['development_end']) else 'late_stress'}))
        # Persist completed origins, so long CPU runs can be inspected independently.
        result = pd.concat(records, ignore_index=True)
        result.to_parquet(out / 'predictions.parquet', index=False)
        print(f'{args.mode}: completed {origin_date}, {len(ids)} municipalities; {time.time()-started:.1f}s',flush=True)
    result = pd.concat(records, ignore_index=True)
    result.to_parquet(out / 'predictions.parquet', index=False)
    metrics = summarize(result)
    metrics.to_csv(out / 'metrics.csv', index=False)
    metadata = dict(vars(args), runtime_seconds=time.time()-started, complete=True)
    (out / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(metrics.to_string(index=False))

if __name__ == '__main__':
    main()
