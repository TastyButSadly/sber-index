"""Matured-label prequential intervals around the reproduced point ensemble.

Calibration targets are separated from scale-model training targets. Evaluation
remains exploratory on the previously inspected, dependent municipal panel.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel

KEYS=['territory_id','origin','date','horizon']
OUT=ROOT/'results/uncertainty'

def state_features(values,origin,target,h,disagreement,dates):
    log=np.log(values[:,:origin+1,target]);factor=np.median(log,axis=0)
    r=log-factor[None,:]
    previous=dates[origin+h]-pd.DateOffset(years=1)
    target_days=dates[origin+h].days_in_month
    calendar_delta=np.log(target_days/previous.days_in_month)
    return np.column_stack([r[:,-1],r[:,-1]-r[:,-2],r[:,-3:].mean(axis=1),
        np.median(np.abs(np.diff(r[:,-6:],axis=1)),axis=1),r[:,origin+h-12],
        log[:,-1],log[:,-1]-log[:,origin-12],disagreement,
        np.full(len(log),h),np.full(len(log),calendar_delta)])

def quantile(scores,coverage):
    scores=np.sort(np.asarray(scores));rank=int(np.ceil((len(scores)+1)*coverage))
    if rank>len(scores):return np.inf
    return scores[rank-1]

def evaluate(pred,coverage):
    rows=[]
    for (model,window,h),g in pred.groupby(['model','window','horizon']):
        covered=(g.actual>=g.lower)&(g.actual<=g.upper)
        width=g.upper-g.lower
        score=width+2/(1-coverage)*(np.maximum(g.lower-g.actual,0)+np.maximum(g.actual-g.upper,0))
        rows.append(dict(model=model,window=window,horizon=h,n=len(g),coverage=covered.mean(),
            median_width_rub=width.median(),mean_interval_score=score.mean(),point_mae=np.abs(g.actual-g.prediction).mean()))
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    cfg=json.loads((ROOT/'configs/uncertainty.json').read_text())
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    source=pd.read_parquet(ROOT/'results/reference_audit/predictions.parquet',filters=[('model','in',
        ['public_ens3_mean','public_ets_relative','public_bolt_base','public_snaive_growth'])])
    base=source[source.model==cfg['point_model']].copy()
    component=source[source.model!=cfg['point_model']].pivot(index=KEYS,columns='model',values='prediction')
    component['disagreement']=np.log(component).std(axis=1)
    base=base.merge(component[['disagreement']],on=KEYS,validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
    features=np.empty((len(base),10))
    for (o,h),g in base.groupby(['origin','horizon']):
        order=g.set_index('territory_id').reindex(ids)
        x=state_features(values,dates.get_loc(o),target,h,order.disagreement.to_numpy(),dates)
        positions=pd.Series(np.arange(len(ids)),index=ids).reindex(g.territory_id).to_numpy()
        features[g.index]=x[positions]
    error=np.log(base.actual/base.prediction).to_numpy()
    abs_error=np.abs(error)
    outputs=[];protocol=[]
    for o in sorted(base.loc[base.origin>=pd.Timestamp(cfg['first_origin']),'origin'].unique()):
        o=pd.Timestamp(o)
        past=(base.date<=o)&(base.origin<o)
        months=sorted(base.loc[past,'date'].unique())
        cut=months[-cfg['calibration_target_months']]
        train=past&(base.date<cut);cal=past&(base.date>=cut);test=base.origin==o
        assert base.loc[past,'date'].max()<=o
        assert set(base.loc[train,'date']).isdisjoint(set(base.loc[cal,'date']))
        if base.loc[train,'date'].nunique()<cfg['minimum_training_target_months']:continue
        scaler=StandardScaler().fit(features[train])
        xtrain=scaler.transform(features[train]);xcal=scaler.transform(features[cal]);xtest=scaler.transform(features[test])
        with threadpool_limits(2):
            signed=BayesianRidge().fit(xtrain,error[train])
            magnitude=BayesianRidge().fit(xtrain,np.log(np.maximum(abs_error[train],cfg['scale_floor_log'])))
        _,std_cal=signed.predict(xcal,return_std=True);_,std_test=signed.predict(xtest,return_std=True)
        scales={
            'global_log':(np.ones(cal.sum()),np.ones(test.sum())),
            'disagreement_log':(np.maximum(base.loc[cal,'disagreement'].to_numpy(),cfg['scale_floor_log']),
                                np.maximum(base.loc[test,'disagreement'].to_numpy(),cfg['scale_floor_log'])),
            'bayesian_predictive':(std_cal,std_test),
            'bayesian_error_scale':(np.exp(magnitude.predict(xcal)),np.exp(magnitude.predict(xtest))),
        }
        for name in cfg['models']:
            sc,st=scales[name];sc=np.maximum(sc,cfg['scale_floor_log']);st=np.maximum(st,cfg['scale_floor_log'])
            for h in sorted(base.loc[test,'horizon'].unique()):
                c=base.loc[cal,'horizon'].to_numpy()==h;t=base.loc[test,'horizon'].to_numpy()==h
                q=quantile(abs_error[cal][c]/sc[c],cfg['nominal_coverage'])
                g=base.loc[test].iloc[np.flatnonzero(t)].copy()
                radius=q*st[t]
                g['lower']=g.prediction*np.exp(-radius);g['upper']=g.prediction*np.exp(radius)
                g['scale_log']=st[t];g['q']=q;g['model']=name
                g['anomaly_score_observed']=np.abs(np.log(g.actual/g.prediction))/st[t]
                # This score requires actuals and must never be an input to its own forecast.
                outputs.append(g)
                protocol.append(dict(origin=str(o.date()),horizon=int(h),model=name,train_n=int(train.sum()),
                    train_target_end=str(base.loc[train,'date'].max().date()),calibration_n=int(c.sum()),
                    calibration_start=str(pd.Timestamp(cut).date()),calibration_target_end=str(base.loc[cal,'date'].max().date()),
                    point_forecast_uses='stored frozen reference model',q=float(q),availability_mode=cfg['availability_mode']))
        print('Completed intervals',o.date(),flush=True)
    predictions=pd.concat(outputs,ignore_index=True)
    predictions['window']=np.where(predictions.origin.between('2024-06-01','2024-09-01'),'public_origins_jun_sep','later_origins')
    predictions.to_parquet(OUT/'predictions.parquet',index=False)
    evaluate(predictions,cfg['nominal_coverage']).to_csv(OUT/'metrics.csv',index=False)
    combined=predictions.copy();combined.horizon='all'
    evaluate(combined,cfg['nominal_coverage']).to_csv(OUT/'aggregate_metrics.csv',index=False)
    pd.DataFrame(protocol).to_csv(OUT/'fold_protocol.csv',index=False)
    by_month=[]
    for (model,date),g in predictions.groupby(['model','date']):
        by_month.append(dict(model=model,target_month=str(date.date()),n=len(g),
            coverage=((g.actual>=g.lower)&(g.actual<=g.upper)).mean(),median_width_rub=(g.upper-g.lower).median()))
    pd.DataFrame(by_month).to_csv(OUT/'monthly_coverage.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
    print(evaluate(combined,cfg['nominal_coverage']).round(3).to_string(index=False))

if __name__=='__main__':main()
