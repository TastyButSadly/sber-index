"""Independent monthly Prophet baseline; this is not the unpublished organizer model."""
import argparse
import json
import numpy as np
import pandas as pd
from prophet import Prophet
from backtest import ROOT, load_panel, summarize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--origins', nargs='+', default=['2024-05-01'])
    parser.add_argument('--output', default='results/prophet_smoke')
    args = parser.parse_args()
    cfg = json.loads((ROOT/'configs/backtest.json').read_text(encoding='utf-8'))
    ids, dates, cats, values = load_panel(args.limit)
    target = cats.index(cfg['target'])
    rows = []
    for origin_date in args.origins:
        origin = dates.get_loc(pd.Timestamp(origin_date))
        if origin < 11:
            raise ValueError('Need 12 observed months')
        hs = [h for h in cfg['horizons'] if origin+h < len(dates)]
        for i, territory in enumerate(ids):
            model = Prophet(yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode='multiplicative', changepoint_prior_scale=0.05, uncertainty_samples=0)
            model.fit(pd.DataFrame({'ds': dates[:origin+1], 'y': values[i, :origin+1, target]}), seed=cfg['seed'])
            forecast = model.predict(pd.DataFrame({'ds': dates[[origin+h for h in hs]]})).yhat.to_numpy()
            for h, pred in zip(hs, forecast):
                rows.append({'territory_id': territory, 'origin': dates[origin], 'date': dates[origin+h],
                    'horizon': h, 'model': 'prophet_monthly', 'prediction': pred,
                    'actual': values[i, origin+h, target], 'seasonal_value': values[i, origin+h-12, target],
                    'split': 'development' if dates[origin+h] <= pd.Timestamp(cfg['development_end']) else 'late_stress'})
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out/'predictions.parquet', index=False)
    summary = summarize(frame)
    summary.to_csv(out/'metrics.csv', index=False)
    (out/'run.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
