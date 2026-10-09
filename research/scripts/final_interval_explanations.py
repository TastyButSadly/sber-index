"""Explain each selected interval and attach future intervals without future labels."""
import json
import numpy as np
import pandas as pd
from backtest import ROOT,load_panel
from final_protocol_intervals import scale_at,finite_quantile

OUT=ROOT/'results/final_project'


def main():
    cfg=json.loads((ROOT/'configs/final_project.json').read_text())
    ids,dates,cats,values=load_panel();total=cats.index('Все категории')
    p=pd.read_parquet(OUT/'predictions.parquet');base=p[p.model==cfg['point_model']]
    selected=pd.read_csv(OUT/'selected_cases.csv',parse_dates=['origin','date'])
    explanations=[]
    for row in selected.itertuples():
        scales,cal,_=scale_at(row.origin,base,cfg,ids,dates,values,total)
        ch=cal[cal.horizon==row.horizon].copy()
        ch['score']=abs(np.log(ch.actual/ch.prediction))/scales[row.horizon].reindex(ch.territory_id).to_numpy()
        q=finite_quantile(ch.score.to_numpy(),.9);scale=float(scales[row.horizon].loc[row.territory_id]);radius=q*scale
        train=base[(base.date<ch.date.min())&(base.origin<row.origin)&(base.horizon==row.horizon)]
        own=train[train.territory_id==row.territory_id]
        individual=float(np.median(abs(np.log(own.actual/own.prediction)))) if len(own) else None
        global_scale=float(np.median(abs(np.log(train.actual/train.prediction)))) if len(train) else None
        near=ch.iloc[np.argsort(abs(ch.score-q))[:3]]
        explanations.append(dict(case_rule=row.case_rule,territory_id=int(row.territory_id),target=str(row.date.date()),
            nominal=.9,prediction_rub=row.prediction,lower_rub=row.lower,upper_rub=row.upper,
            scale_log=scale,quantile=q,log_radius=radius,lower_multiplier=float(np.exp(-radius)),upper_multiplier=float(np.exp(radius)),
            calibration_start=str(ch.date.min().date()),calibration_end=str(ch.date.max().date()),calibration_n=len(ch),
            quantile_rank=int(np.ceil((len(ch)+1)*.9)),training_municipal_errors=len(own),
            municipal_old_median_abs_log_error=individual,horizon_old_median_abs_log_error=global_scale,
            municipal_scale_weight=len(own)/(len(own)+cfg['scale_shrinkage_pseudocount']),
            nearest_quantile_observations=[dict(territory_id=int(r.territory_id),origin=str(r.origin.date()),target=str(r.date.date()),score=float(r.score)) for r in near.itertuples()],
            explanation='Bounds = frozen point forecast multiplied by exp(plus/minus empirical quantile times municipality scale). Scale uses older errors, quantile uses separate recent matured errors.',
            interpretation='Prediction interval for the published municipal estimate; not a confidence interval for the mean or total consumption of every resident. Marginal empirical coverage, no municipal conditional guarantee.'))
        assert np.isclose(row.lower,row.prediction*np.exp(-radius)) and np.isclose(row.upper,row.prediction*np.exp(radius))
    (OUT/'case_interval_explanations.json').write_text(json.dumps(explanations,ensure_ascii=False,indent=2),encoding='utf-8')
    future=pd.read_parquet(ROOT/'results/combined_sources/future_forecasts.parquet')
    future=future[future.model==cfg['point_model']+'_ema'].copy();future['model']=cfg['point_model']
    variants={cfg['point_model']:future.prediction.to_numpy(),
        'component_median':np.median(future[['bolt_prediction','ridge_prediction','seasonal_prediction']].to_numpy(),axis=1),
        'ridge_light':future[['bolt_prediction','ridge_prediction','seasonal_prediction']].to_numpy()@np.array(cfg['ridge_light_weights_bolt_ridge_seasonal']),
        'seasonal_blend':.5*future.prediction.to_numpy()+.5*future.seasonal_prediction.to_numpy()}
    rows=[]
    for name,forecast in variants.items():
        g=future.copy();g['model']=name;g['prediction']=forecast;rows.append(g)
    pd.concat(rows,ignore_index=True).to_parquet(OUT/'future_forecasts.parquet',index=False)
    scales,cal,_=scale_at(dates[-1],base,cfg,ids,dates,values,total);rows=[]
    for method in cfg['interval_methods']:
        for h,g in future.groupby('horizon'):
            ch=cal if method=='global_log' else cal[cal.horizon==h]
            cs=scales[h].reindex(ch.territory_id).to_numpy() if method=='adaptive_log' else np.ones(len(ch))
            scale=scales[h].reindex(g.territory_id).to_numpy() if method=='adaptive_log' else np.ones(len(g))
            scores=abs(np.log(ch.actual/ch.prediction)).to_numpy()/cs
            for nominal in cfg['nominal_levels']:
                q=finite_quantile(scores,nominal);d=g.copy();d['interval_method']=method;d['nominal']=nominal;d['q']=q;d['scale_log']=scale
                d['lower']=d.prediction*np.exp(-q*scale);d['upper']=d.prediction*np.exp(q*scale)
                d['calibration_start']=ch.date.min();d['calibration_end']=ch.date.max();d['calibration_n']=len(ch)
                rows.append(d)
    pd.concat(rows,ignore_index=True).to_parquet(OUT/'future_interval_forecasts.parquet',index=False)
    manifest=pd.read_parquet(OUT/'forecast_manifest.parquet')
    if 'purpose' not in manifest:manifest['purpose']='evaluation'
    extra=future[['territory_id','origin','date','horizon']].rename(columns={'date':'target'}).copy()
    extra['last_observed_month']=extra.origin;extra['asof_observation_time']=extra.origin+pd.offsets.MonthEnd(0)
    extra['verified_release_asof']=pd.NaT;extra['point_model']=cfg['point_model'];extra['purpose']='future'
    manifest=pd.concat([manifest[manifest.purpose=='evaluation'],extra],ignore_index=True)
    assert not manifest.duplicated(['territory_id','origin','target','horizon']).any()
    manifest.to_parquet(OUT/'forecast_manifest.parquet',index=False)
    (OUT/'future_manifest.json').write_text(json.dumps(dict(origin=str(dates[-1].date()),target_months=['2025-01','2025-02','2025-03'],
        point_model=cfg['point_model'],robust_variants_preserved=True,interval_model='frozen baseline only; robust variants are point ablations',
        labels_2025_available=False,coverage_2025='unknown',prediction_bounds_explained=True,strict_real_time_eligible=False),indent=2))
    print('Four case interval explanations and 2025 conditional interval forecasts saved.')


if __name__=='__main__':main()
