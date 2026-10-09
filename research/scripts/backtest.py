"""Expanding-window backtest. No future observations enter model inputs or fitting."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

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

def growth_forecast(values, origin, h, target, window):
    # Only YoY changes available at the forecast origin are used.
    start = max(12, origin - window + 1)
    if origin < 12:
        return values[:, origin+h-12, target].copy()
    growth = np.median(np.log(values[:, start:origin+1, target] / values[:, start-12:origin-11, target]), axis=1)
    return values[:, origin+h-12, target] * np.exp(growth)

def panel_forecast(values, origin, h, target, cfg):
    observed = np.log(values[:, :origin+1, :])
    factor = np.median(observed, axis=0)
    residual = observed - factor[None, :, :]
    def features(s):
        return np.concatenate([residual[:, s], residual[:, s-1], residual[:, s+h-12], residual[:, s]-residual[:, s-1]], axis=1)
    train_origins = list(range(max(1, 12-h), origin-h+1))
    if len(train_origins) < cfg['min_training_origins']:
        return None, len(train_origins)
    x = np.concatenate([features(s) for s in train_origins])
    y = np.concatenate([residual[:, s+h, target] for s in train_origins])
    weights = np.concatenate([(values[:, s+h-12, target] / np.median(values[:, s+h-12, target])) ** cfg['weight_power'] for s in train_origins])
    weights /= weights.mean()
    model = Ridge(alpha=cfg['ridge_alpha']).fit(x, y, sample_weight=weights)
    start = max(12, origin-cfg['growth_window']+1)
    factor_growth = np.median(factor[start:origin+1, target] - factor[start-12:origin-11, target])
    future_factor = factor[origin+h-12, target] + factor_growth
    return np.exp(model.predict(features(origin)) + future_factor), len(train_origins)

def summarize(pred):
    rows = []
    for keys, group in pred.groupby(['split', 'horizon', 'model']):
        actual, predicted = group.actual.to_numpy(), group.prediction.to_numpy()
        prior = group.seasonal_value.to_numpy()
        rows.append(dict(zip(['split', 'horizon', 'model'], keys), n=len(group), origins=group.origin.nunique(),
                         mae_rub=mean_absolute_error(actual, predicted), r2_level=r2_score(actual, predicted),
                         r2_yoy=r2_score(actual/prior-1, predicted/prior-1)))
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/backtest.json')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output', default='results/baselines')
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    ids, dates, cats, values = load_panel(args.limit)
    target = cats.index(cfg['target'])
    records, availability = [], []
    for origin, date in enumerate(dates):
        if origin < 11 or date < pd.Timestamp(cfg['first_origin']):
            continue
        for h in cfg['horizons']:
            if origin+h >= len(dates):
                continue
            seasonal = values[:, origin+h-12, target]
            forecasts = {'naive': values[:, origin, target], 'seasonal_naive': seasonal,
                         'seasonal_growth': growth_forecast(values, origin, h, target, cfg['growth_window'])}
            ridge, count = panel_forecast(values, origin, h, target, cfg)
            availability.append({'origin': str(date.date()), 'horizon': h, 'training_origins': count, 'ridge_available': ridge is not None})
            if ridge is not None:
                forecasts['weighted_panel_ridge'] = ridge
                forecasts['ridge_seasonal_ensemble'] = cfg['panel_weight']*ridge+(1-cfg['panel_weight'])*forecasts['seasonal_growth']
            split = 'development' if dates[origin+h] <= pd.Timestamp(cfg['development_end']) else 'late_stress'
            for model, forecast in forecasts.items():
                records.append(pd.DataFrame({'territory_id': ids, 'origin': date, 'date': dates[origin+h],
                    'horizon': h, 'model': model, 'prediction': forecast, 'actual': values[:, origin+h, target],
                    'seasonal_value': seasonal, 'split': split}))
    predictions = pd.concat(records, ignore_index=True)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out / 'predictions.parquet', index=False)
    summary = summarize(predictions)
    summary.to_csv(out / 'metrics.csv', index=False)
    # Pair comparisons on exactly the origins where Ridge could be trained.
    eligible = predictions[predictions.model == 'weighted_panel_ridge'][['origin', 'horizon']].drop_duplicates()
    paired = predictions.merge(eligible, on=['origin', 'horizon'])
    summarize(paired).to_csv(out / 'paired_metrics.csv', index=False)
    pd.DataFrame(availability).to_csv(out / 'availability.csv', index=False)
    (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
