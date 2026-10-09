"""Frozen point protocol, fixed robust combinations and explained prequential intervals."""
import json
import hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from backtest import ROOT,load_panel
from develop_point_ensemble import factor_forecast

KEYS=['territory_id','origin','date','horizon']
OUT=ROOT/'results/final_project'


def finite_quantile(values,level):
    a=np.sort(values);rank=int(np.ceil((len(a)+1)*level))
    if rank>len(a):return np.inf
    return float(a[rank-1])


def scale_at(origin,base,cfg,ids,dates,values,total):
    past=base[(base.date<=origin)&(base.origin<origin)]
    months=sorted(past.date.unique())
    cut=pd.Timestamp(months[-cfg['calibration_target_months']])
    train=past[past.date<cut];cal=past[past.date>=cut]
    train=train.copy();train['abs_log_error']=abs(np.log(train.actual/train.prediction))
    i=dates.get_loc(origin)
    local=np.log(values[:,:i+1,total]);local-=np.median(local,axis=0)[None,:]
    # Past local variability is the fallback when a municipality has no matured training errors.
    differences=np.diff(local[:,-7:],axis=1)
    fallback=np.maximum(np.median(abs(differences-np.median(differences,axis=1)[:,None]),axis=1)*1.4826,cfg['scale_floor_log'])
    result={};audit=[]
    for h in cfg['horizons']:
        tr=train[train.horizon==h];group=tr.groupby('territory_id').abs_log_error.agg(['median','count']).reindex(ids)
        pooled=float(tr.abs_log_error.median()) if len(tr) else float(np.median(fallback)*np.sqrt(h))
        n=group['count'].fillna(0).to_numpy();individual=group['median'].fillna(pd.Series(fallback*np.sqrt(h),index=ids)).to_numpy()
        weight=n/(n+cfg['scale_shrinkage_pseudocount'])
        scale=np.maximum(weight*individual+(1-weight)*pooled,cfg['scale_floor_log'])
        result[h]=pd.Series(scale,index=ids)
        audit.append(dict(origin=str(origin.date()),horizon=h,training_end=str(tr.date.max().date()) if len(tr) else None,
            calibration_start=str(cut.date()),calibration_end=str(cal.date.max().date()),
            global_train_median_log_error=pooled,training_months=int(tr.date.nunique()),
            interpretation='municipal median of older matured absolute log errors, shrunk to horizon median; no calibration labels used in scale fit'))
    return result,cal,audit


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/final_project.json').read_text())
    source=ROOT/'results/recomputed_base.parquet'
    base=pd.read_parquet(source).sort_values(KEYS).reset_index(drop=True)
    assert not base.duplicated(KEYS).any()
    (OUT/'protocol.json').write_text(json.dumps(dict(config=cfg,started_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        decision_time='origin month end in observed-prefix simulation; actual publication cutoff unverified'),indent=2))
    manifest=base[KEYS].rename(columns={'date':'target'}).copy()
    manifest['last_observed_month']=manifest.origin;manifest['asof_observation_time']=manifest.origin+pd.offsets.MonthEnd(0)
    manifest['verified_release_asof']=pd.NaT;manifest['point_model']=cfg['point_model']
    manifest.to_parquet(OUT/'forecast_manifest.parquet',index=False)
    base[KEYS+['actual']].to_parquet(OUT/'evaluation_truth.parquet',index=False)
    ids,dates,cats,values=load_panel();total=cats.index('Все категории')
    factor=np.median(np.log(values[:,:,total]),axis=0)
    components=pd.read_parquet(ROOT/'results/recomputed_components.parquet').set_index(KEYS).reindex(pd.MultiIndex.from_frame(base[KEYS])).reset_index()
    bolt=components.bolt.to_numpy();ridge=components.ridge.to_numpy();seasonal=components.seasonal.to_numpy()
    assert np.allclose(base.prediction,(bolt+ridge+seasonal)/3)
    components.to_parquet(OUT/'components.parquet',index=False)
    variants={cfg['point_model']:base.prediction.to_numpy(),
              'component_median':np.median(np.c_[bolt,ridge,seasonal],axis=1),
              'ridge_light':np.c_[bolt,ridge,seasonal]@np.array(cfg['ridge_light_weights_bolt_ridge_seasonal']),
              'seasonal_blend':(1-cfg['seasonal_blend_weight'])*base.prediction.to_numpy()+cfg['seasonal_blend_weight']*seasonal}
    rows=[]
    for name,p in variants.items():
        d=base.copy();d['model']=name;d['prediction']=p;rows.append(d)
    point=pd.concat(rows,ignore_index=True);point.to_parquet(OUT/'predictions.parquet',index=False)
    segments=[]
    for origin,g in base.groupby('origin'):
        o=dates.get_loc(origin);level=values[:,o,total]
        local=np.log(values[:,:o+1,total]);local-=np.median(local,axis=0)[None,:]
        vol=np.std(np.diff(local[:,-7:],axis=1),axis=1)
        frame=pd.DataFrame(dict(territory_id=ids,origin=origin,
            spending_quintile=pd.qcut(pd.Series(level).rank(method='first'),5,labels=False)+1,
            volatility_quintile=pd.qcut(pd.Series(vol).rank(method='first'),5,labels=False)+1))
        segments.append(frame)
    segments=pd.concat(segments,ignore_index=True);segments.to_parquet(OUT/'origin_segments.parquet',index=False)
    diagnostic=point.merge(segments,on=['territory_id','origin'],validate='many_to_one')
    diagnostic['window']=np.where(diagnostic.origin.between('2024-06-01','2024-09-01'),'public_origins_jun_sep',
                                 np.where(diagnostic.origin>pd.Timestamp('2024-09-01'),'later_origins','earlier'))
    diagnostic['absolute_error']=abs(diagnostic.actual-diagnostic.prediction);diagnostic['error']=diagnostic.prediction-diagnostic.actual
    for suffix,cols in [('window',['window']),('horizon',['window','horizon']),('month',['date','horizon']),
                        ('spending',['window','spending_quintile']),('volatility',['window','volatility_quintile'])]:
        diagnostic.groupby(['model']+cols,as_index=False).agg(n=('actual','size'),mae_rub=('absolute_error','mean'),bias_rub=('error','mean')).to_csv(OUT/f'{suffix}_metrics.csv',index=False)
    window=diagnostic[diagnostic.window!='earlier'].groupby(['model','window']).absolute_error.mean().unstack()
    window['late_minus_main']=window.later_origins-window.public_origins_jun_sep
    window['late_over_main']=window.later_origins/window.public_origins_jun_sep
    monthly=diagnostic[diagnostic.window!='earlier'].groupby(['model','date']).absolute_error.mean().groupby('model').max()
    window['worst_target_month_mae']=monthly;window.to_csv(OUT/'degradation.csv')
    outputs=[];folds=[]
    for origin in sorted(base.loc[base.origin>=pd.Timestamp(cfg['interval_first_origin']),'origin'].unique()):
        origin=pd.Timestamp(origin);scales,cal,audit=scale_at(origin,base,cfg,ids,dates,values,total);folds+=audit
        assert (cal.date<=origin).all()
        for method in cfg['interval_methods']:
            for h,g in base[base.origin==origin].groupby('horizon'):
                calibration=cal if method=='global_log' else cal[cal.horizon==h]
                if method=='adaptive_log':
                    cs=scales[h].reindex(calibration.territory_id).to_numpy();scale=scales[h].reindex(g.territory_id).to_numpy()
                else:cs=np.ones(len(calibration));scale=np.ones(len(g))
                scores=abs(np.log(calibration.actual/calibration.prediction)).to_numpy()/cs
                for nominal in cfg['nominal_levels']:
                    q=finite_quantile(scores,nominal);assert np.isfinite(q)
                    d=g.copy();d['interval_method']=method;d['nominal']=nominal;d['scale_log']=scale;d['q']=q
                    d['lower']=d.prediction*np.exp(-q*scale);d['upper']=d.prediction*np.exp(q*scale)
                    d['calibration_n']=len(scores);d['calibration_months']=calibration.date.nunique()
                    d['calibration_start']=calibration.date.min();d['calibration_end']=calibration.date.max()
                    d['quantile_rank']=int(np.ceil((len(scores)+1)*nominal));outputs.append(d)
        print('Final intervals',origin.date(),flush=True)
    interval=pd.concat(outputs,ignore_index=True);interval.to_parquet(OUT/'interval_predictions.parquet',index=False)
    interval['covered']=(interval.actual>=interval.lower)&(interval.actual<=interval.upper)
    interval['width_rub']=interval.upper-interval.lower
    interval['interval_score']=interval.width_rub+2/(1-interval.nominal)*(np.maximum(interval.lower-interval.actual,0)+np.maximum(interval.actual-interval.upper,0))
    interval['window']=np.where(interval.origin.between('2024-06-01','2024-09-01'),'public_origins_jun_sep','later_origins')
    for suffix,cols in [('window',['window']),('horizon',['window','horizon']),('month',['date'])]:
        interval.groupby(['interval_method','nominal']+cols,as_index=False).agg(n=('actual','size'),coverage=('covered','mean'),
            mean_width_rub=('width_rub','mean'),median_width_rub=('width_rub','median'),interval_score=('interval_score','mean')).to_csv(OUT/f'interval_{suffix}_metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'interval_folds.csv',index=False)
    origin=pd.Timestamp('2024-08-01');bad=base.copy();bad.loc[bad.date>origin,'actual']*=1000
    changed=values.copy();changed[:,dates>origin]*=1000
    a,cal,_=scale_at(origin,base,cfg,ids,dates,values,total);b,other,_=scale_at(origin,bad,cfg,ids,dates,changed,total)
    for h in cfg['horizons']:assert np.array_equal(a[h],b[h])
    assert np.array_equal(cal.actual,other.actual)
    assert np.allclose(interval.lower,interval.prediction*np.exp(-interval.q*interval.scale_log))
    assert np.allclose(interval.upper,interval.prediction*np.exp(interval.q*interval.scale_log))
    (OUT/'checks.json').write_text(json.dumps(dict(status='passed',unique_manifest=True,
        frozen_point_forecasts=True,scale_and_calibration_targets_disjoint=True,future_labels_and_values_excluded=True,
        interval_bounds_explained=True,population_size_unavailable=True,strict_real_time_eligible=False,
        reconstruction='explicit Bolt, extended Ridge and seasonal components on identical EMA factor'),indent=2))
    print(window.to_string(),flush=True)


if __name__=='__main__':main()
