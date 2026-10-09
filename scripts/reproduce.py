"""Recompute final forecasts, intervals and detection from frozen input snapshots."""
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from backtest import ROOT, load_panel
from forecast_point import PointEnsemble
from panel_features import dataset
from architecture_extensions import structural_features, KEYS
from develop_point_ensemble import factor_forecast


def fit_origin(origin):
    ids, dates, cats, values = load_panel()
    i = dates.get_loc(pd.Timestamp(origin))
    horizons = [h for h in (1, 2, 3) if i+h < len(dates)]
    observed = values[:, :i+1]
    cfg = json.loads((ROOT/'configs/combined_sources.json').read_text())
    target = cats.index('Все категории')
    point = PointEnsemble(horizons).fit(observed, ids, dates[:i+1], cats)
    components = point.predict()
    components = components[components.model == 'bolt_ridge_seasonal_ema'].copy()
    logs = np.log(observed)
    factor = np.median(logs, axis=0)
    residual = logs-factor[None]
    macro = pd.read_parquet(ROOT/'data/external/macro/monthly_features.parquet').reindex(dates[:i+1]).to_numpy()
    geo = pd.read_parquet(ROOT/'data/processed/municipalities_2024.parquet').drop_duplicates('territory_id').set_index('territory_id')
    regions = geo.region_code.reindex(ids).to_numpy()
    regional = pd.read_parquet(ROOT/'data/external/regional/features.parquet').set_index(['region_code', 'date'])
    columns = regional.columns.tolist()
    selected = [j for j,c in enumerate(columns) if c.startswith(('cpi_', 'wage_'))]
    r = regional.reindex(pd.MultiIndex.from_product([regions, dates[:i+1]])).to_numpy().reshape(len(ids), i+1, len(columns))

    def extra(s, h):
        structure = structural_features(observed[:, :s+1], residual[:, :s+1], s, h, target)*cfg['structure_scale']
        mf = np.tile(macro[s], (len(ids), 1))
        shares = observed[:, s, [cats.index('Маркетплейсы'), cats.index('Продовольствие')]]/observed[:, s, target, None]
        shares -= np.median(shares, axis=0)
        national = np.c_[mf, mf*shares[:, 0, None], mf*shares[:, 1, None]]*cfg['national_macro_scale']
        return np.c_[structure, national, r[:, s, selected]*cfg['regional_scale']]

    frames = []
    for h in horizons:
        x, y, scale, xf, starts = dataset(observed, residual, i, h, target, np.arange(len(cats)))
        xx = np.concatenate([extra(s, h) for s in starts])
        weights = scale**cfg['sample_weight_power']; weights /= weights.mean()
        with threadpool_limits(2):
            fit = Ridge(alpha=cfg['ridge_alpha']).fit(np.c_[x, xx], y, sample_weight=weights)
        g = components[components.horizon == h].set_index('territory_id').reindex(ids).reset_index()
        g['ridge_prediction'] = np.exp(fit.predict(np.c_[xf, extra(i, h)])+factor_forecast(factor[:, target], i, h, 'ema'))
        g['actual'] = values[:, i+h, target]
        g['seasonal_value'] = values[:, i+h-12, target]
        frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    c = g[KEYS+['bolt_prediction', 'ridge_prediction', 'seasonal_prediction']].rename(columns={n+'_prediction':n for n in ('bolt','ridge','seasonal')})
    base = g[KEYS+['actual', 'seasonal_value']].copy()
    base['model'] = 'structure_macro_prices_wages'
    base['prediction'] = c[['bolt','ridge','seasonal']].mean(axis=1).to_numpy()
    return base, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['cached', 'fit'], default='cached')
    p.add_argument('--origin', help='Optional single origin for a fit check; full protocol needs all origins')
    p.add_argument('--detection', action='store_true')
    args = p.parse_args()
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    # Write the fixed configuration before fitting/scoring.
    (out/'reproduction_protocol.json').write_text(json.dumps(dict(mode=args.mode,origin=args.origin,seed=20261009,config=json.loads((ROOT/'configs/final_project.json').read_text()),inputs='frozen observed-prefix snapshots'),ensure_ascii=False,indent=2),encoding='utf-8')
    saved = pd.read_parquet(ROOT/'evidence/final_project/predictions.parquet')
    saved = saved[saved.model == 'structure_macro_prices_wages'].sort_values(KEYS).reset_index(drop=True)
    if args.mode == 'cached':
        base = saved
        components = pd.read_parquet(ROOT/'evidence/final_project/components.parquet')
    else:
        origins = [pd.Timestamp(args.origin)] if args.origin else sorted(saved.origin.unique())
        bases, parts = [], []
        for origin in origins:
            b, c = fit_origin(origin); bases.append(b); parts.append(c)
            print('Fitted', str(pd.Timestamp(origin).date()), len(b), 'forecasts', flush=True)
        base = pd.concat(bases, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        components = pd.concat(parts, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        match = base.merge(saved[KEYS+['prediction']], on=KEYS, suffixes=('', '_saved'), validate='one_to_one')
        error = float(abs(match.prediction-match.prediction_saved).max())
        print('Maximum difference from frozen equal-weight forecasts, RUB:', error, flush=True)
        if error > 0.10:
            raise ValueError('Forecast reproduction differs by more than 0.10 RUB; inspect environment and inputs')
    base.to_parquet(out/'recomputed_base.parquet', index=False)
    components.to_parquet(out/'recomputed_components.parquet', index=False)
    if args.origin:
        print('Single-origin verification finished; full evaluation was not run.', flush=True)
        return
    subprocess.run([sys.executable, str(ROOT/'scripts/final_protocol_intervals.py')], check=True)
    if args.detection:
        subprocess.run([sys.executable, str(ROOT/'scripts/final_detection_benchmark.py')], check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/verify.py')], check=True)


if __name__ == '__main__':
    main()
