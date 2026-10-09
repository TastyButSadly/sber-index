"""Reuse audited forecasts for a documented horizon portfolio, without model reloads."""
import json
import pandas as pd
from backtest import ROOT


def main():
    out=ROOT/'results/contest_horizons'
    cfg=json.loads((ROOT/'configs/contest_comparison.json').read_text())
    sample=pd.read_csv(out/'sample.csv').territory_id
    short=pd.read_parquet(ROOT/'results/point_ensemble/all_candidate_predictions.parquet',filters=[('model','==','bolt_ridge_seasonal_ema')])
    old=pd.read_parquet(ROOT/'results/hypotheses/predictions.parquet',filters=[('model','==','ridge_seasonal_ensemble')])
    early=pd.read_parquet(out/'forecast_checkpoint.parquet')
    early=early[early.model=='bolt_seasonal_fallback']
    frames=[]
    for raw_o in cfg['origins']:
        o=pd.Timestamp(raw_o)
        for h in cfg['horizons']:
            if o+pd.DateOffset(months=h)>pd.Timestamp('2024-12-01'):continue
            if o==pd.Timestamp('2023-12-01'):
                g=early[(early.origin==o)&(early.horizon==h)].copy();g['model']='portfolio_early_bolt_seasonal'
            elif h in [1,3]:
                g=short[(short.origin==o)&(short.horizon==h)].copy();g['model']='portfolio_short_ema'
            elif h==6:
                g=old[(old.origin==o)&(old.horizon==h)].copy();g['model']='portfolio_h6_ridge_seasonal'
            else:raise ValueError('No cached audited model for this fold')
            g=g[g.territory_id.isin(sample)]
            assert len(g)==len(sample),(raw_o,h,len(g))
            frames.append(g[['territory_id','origin','date','horizon','model','prediction']])
    result=pd.concat(frames,ignore_index=True)
    assert not result.duplicated(['territory_id','origin','date','horizon']).any()
    result.to_parquet(out/'ensemble_checkpoint.parquet',index=False)
    (out/'portfolio_protocol.json').write_text(json.dumps(dict(
        short='Bolt + pooled Ridge + seasonal, EMA shared factor',
        h6='previous pooled Ridge + seasonal EMA, equal weights',
        early='December 2023: frozen Bolt relative + seasonal, equal weights; no matured Ridge pairs',
        sources=['results/point_ensemble/all_candidate_predictions.parquet','results/hypotheses/predictions.parquet','results/contest_horizons/forecast_checkpoint.parquet'],
        warning='horizon-dependent exploratory portfolio; not evaluation of the three-component Bolt model at all horizons; parameters previously inspected'),indent=2))
    print('Prepared cached portfolio',len(result),flush=True)


if __name__=='__main__':main()
