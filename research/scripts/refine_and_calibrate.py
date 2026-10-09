"""Matured-label point corrections and empirical intervals for the new ensemble."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,BayesianRidge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from run_uncertainty import state_features,quantile

OUT=ROOT/'results/new_model_calibration'
KEYS=['territory_id','origin','date','horizon']
POINTS=['bolt_ridge_seasonal_mean1','bolt_ridge_seasonal_ema']


def matured_masks(d,o):
    past=(d.date<=o)&(d.origin<o)
    months=sorted(d.loc[past,'date'].unique())
    if len(months)<4:return None
    cut=pd.Timestamp(months[-2])
    return past,past&(d.date<cut),past&(d.date>=cut)


def interval_summary(g,nominal):
    inside=(g.actual>=g.lower)&(g.actual<=g.upper)
    width=g.upper-g.lower
    score=width+2/(1-nominal)*(np.maximum(g.lower-g.actual,0)+np.maximum(g.actual-g.upper,0))
    return dict(n=len(g),coverage=float(inside.mean()),median_width_rub=float(width.median()),
                mean_width_rub=float(width.mean()),interval_score=float(score.mean()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    source=pd.read_parquet(ROOT/'results/point_ensemble/all_candidate_predictions.parquet')
    interval_frames=[];point_frames=[];protocol=[]
    for point in POINTS:
        d=source[source.model==point].sort_values(KEYS).reset_index(drop=True).copy()
        x=np.empty((len(d),10));size=np.zeros(len(d),dtype=int)
        for (o,h),g in d.groupby(['origin','horizon']):
            i=dates.get_loc(o)
            features=state_features(values,i,target,h,np.zeros(len(ids)),dates)
            pos=pd.Series(np.arange(len(ids)),index=ids).reindex(g.territory_id).to_numpy()
            x[g.index]=features[pos]
            # Terciles depend on current observed spending, not future actuals.
            cuts=np.quantile(features[:,5],[1/3,2/3])
            size[g.index]=np.digitize(features[pos,5],cuts)
        d['size_group']=size
        log_error=np.log(d.actual/d.prediction).to_numpy()
        rub_error=np.abs(d.actual-d.prediction).to_numpy()
        modified={name:d.prediction.to_numpy().copy() for name in ['original','log_bias_half','ridge_correction_quarter','ridge_correction_half']}
        for raw_o in sorted(d.loc[d.origin>=pd.Timestamp('2024-06-01'),'origin'].unique()):
            o=pd.Timestamp(raw_o);masks=matured_masks(d,o)
            if masks is None:continue
            past,train,cal=masks;test=d.origin==o
            assert d.loc[cal,'date'].max()<=o
            assert set(d.loc[train,'date']).isdisjoint(set(d.loc[cal,'date']))
            protocol.append(dict(point_model=point,origin=str(o.date()),
                train_target_end=str(d.loc[train,'date'].max().date()),
                calibration_start=str(d.loc[cal,'date'].min().date()),
                calibration_end=str(d.loc[cal,'date'].max().date()),
                train_n=int(train.sum()),calibration_n=int(cal.sum())))
            # Point correction fit on last four matured target months.
            months=sorted(d.loc[past,'date'].unique());recent=past&d.date.isin(months[-4:])
            scaler=StandardScaler().fit(x[recent])
            with threadpool_limits(2):
                correction=Ridge(alpha=100).fit(scaler.transform(x[recent]),log_error[recent])
            change=np.clip(correction.predict(scaler.transform(x[test])),-.1,.1)
            for weight,name in [(.25,'ridge_correction_quarter'),(.5,'ridge_correction_half')]:
                modified[name][test]=d.loc[test,'prediction'].to_numpy()*np.exp(weight*change)
            for h in sorted(d.loc[test,'horizon'].unique()):
                ph=recent&(d.horizon==h);th=test&(d.horizon==h)
                bias=np.clip(np.median(log_error[ph]),-.05,.05)
                modified['log_bias_half'][th]=d.loc[th,'prediction'].to_numpy()*np.exp(.5*bias)
            # Conditional scale: training targets precede calibration targets.
            scaler_scale=StandardScaler().fit(x[train])
            with threadpool_limits(2):
                magnitude=BayesianRidge().fit(scaler_scale.transform(x[train]),np.log(np.maximum(np.abs(log_error[train]),.005)))
            scale_cal=np.maximum(np.exp(magnitude.predict(scaler_scale.transform(x[cal]))),.005)
            scale_test=np.maximum(np.exp(magnitude.predict(scaler_scale.transform(x[test]))),.005)
            for nominal in [.8,.9,.95]:
                for h in sorted(d.loc[test,'horizon'].unique()):
                    c=d.loc[cal,'horizon'].to_numpy()==h;t=d.loc[test,'horizon'].to_numpy()==h
                    g=d.loc[test].iloc[np.flatnonzero(t)].copy()
                    cal_h=cal&(d.horizon==h)
                    global_q=quantile(np.abs(log_error[cal_h]),nominal)
                    size_q={s:quantile(np.abs(log_error[cal_h&(d.size_group==s)]),nominal) for s in range(3)}
                    radii={
                        'global_log':np.full(len(g),global_q),
                        'conditional_log':quantile(np.abs(log_error[cal][c])/scale_cal[c],nominal)*scale_test[t],
                        'size_group_log':np.array([size_q[s] for s in g.size_group]),
                    }
                    for method,radius in radii.items():
                        frame=g.copy();frame['lower']=frame.prediction*np.exp(-radius);frame['upper']=frame.prediction*np.exp(radius)
                        frame['interval_method']=method;frame['nominal']=nominal;interval_frames.append(frame)
                    qr=quantile(rub_error[cal_h],nominal)
                    frame=g.copy();frame['lower']=np.maximum(0,frame.prediction-qr);frame['upper']=frame.prediction+qr
                    frame['interval_method']='global_rub';frame['nominal']=nominal;interval_frames.append(frame)
        for method,pred in modified.items():
            g=d[d.origin>=pd.Timestamp('2024-06-01')].copy()
            g['prediction']=pred[g.index];g['point_method']=method;point_frames.append(g)
    intervals=pd.concat(interval_frames,ignore_index=True);points=pd.concat(point_frames,ignore_index=True)
    for frame in (intervals,points):
        frame['window']=np.where(frame.origin<=pd.Timestamp('2024-09-01'),'public_origins_jun_sep','later_origins')
    intervals.to_parquet(OUT/'interval_predictions.parquet',index=False)
    points.to_parquet(OUT/'corrected_point_predictions.parquet',index=False)
    protocol=pd.DataFrame(protocol);protocol.to_csv(OUT/'protocol.csv',index=False)
    metrics=[]
    for (model,method,window),g in points.groupby(['model','point_method','window']):
        metrics.append(dict(model=model,point_method=method,window=window,n=len(g),mae_rub=float(np.abs(g.actual-g.prediction).mean())))
    pd.DataFrame(metrics).to_csv(OUT/'point_metrics.csv',index=False)
    aggregate=[];horizons=[];monthly=[];groups=[];cis=[]
    rng=np.random.default_rng(20261009)
    for (model,method,nominal,window),g in intervals.groupby(['model','interval_method','nominal','window']):
        meta=dict(model=model,interval_method=method,nominal=nominal,window=window)
        aggregate.append(dict(**meta,**interval_summary(g,nominal)))
        for h,sub in g.groupby('horizon'):horizons.append(dict(**meta,horizon=h,**interval_summary(sub,nominal)))
        for month,sub in g.groupby('date'):monthly.append(dict(**meta,target_month=str(month.date()),**interval_summary(sub,nominal)))
        for s,sub in g.groupby('size_group'):groups.append(dict(**meta,size_group=s,**interval_summary(sub,nominal)))
        summary=g.assign(covered=(g.actual>=g.lower)&(g.actual<=g.upper)).groupby('date').agg(covered=('covered','sum'),n=('covered','size'))
        draw=rng.integers(0,len(summary),size=(2000,len(summary)))
        cov=summary.covered.to_numpy()[draw].sum(axis=1)/summary.n.to_numpy()[draw].sum(axis=1)
        lo,hi=np.quantile(cov,[.025,.975])
        cis.append(dict(**meta,month_clusters=len(summary),coverage_ci_low=lo,coverage_ci_high=hi))
    for name,rows in [('interval_metrics',aggregate),('horizon_coverage',horizons),('monthly_coverage',monthly),('size_group_coverage',groups),('coverage_bootstrap',cis)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    # Fresh future bands around deployed points: only h=1,2,3 have audited calibration.
    future=pd.read_parquet(ROOT/'results/point_forecast/predictions.parquet')
    outputs=[]
    for point in POINTS:
        history=source[source.model==point]
        for (origin,h),g in future[(future.model==point)&future.horizon.isin([1,2,3])].groupby(['origin','horizon']):
            past=history[(history.date<=origin)&(history.origin<origin)&(history.horizon==h)]
            months=sorted(past.date.unique());cal=past[past.date.isin(months[-2:])]
            assert cal.date.max()<=origin
            for nominal in [.8,.9,.95]:
                radius=quantile(np.abs(np.log(cal.actual/cal.prediction)),nominal)
                frame=g.copy();frame['lower']=frame.prediction*np.exp(-radius);frame['upper']=frame.prediction*np.exp(radius)
                frame['nominal']=nominal;frame['interval_method']='global_log';outputs.append(frame)
    pd.concat(outputs,ignore_index=True).to_csv(OUT/'future_intervals.csv',index=False,encoding='utf-8-sig')
    assert np.isfinite(intervals[['prediction','lower','upper']].to_numpy()).all()
    assert ((intervals.lower<=intervals.prediction)&(intervals.upper>=intervals.prediction)).all()
    # Perturb unknown labels: neither training nor calibration subsets change.
    o=pd.Timestamp('2024-09-01');d=source[source.model==POINTS[0]].reset_index(drop=True)
    masks=matured_masks(d,o);bad=d.copy();bad.loc[bad.date>o,'actual']*=1000
    for mask in masks:pd.testing.assert_frame_equal(d.loc[mask],bad.loc[mask])
    for nominal in [.8,.9,.95]:
        cal=masks[2]
        assert quantile(np.abs(np.log(d.loc[cal,'actual']/d.loc[cal,'prediction'])),nominal)==quantile(np.abs(np.log(bad.loc[cal,'actual']/bad.loc[cal,'prediction'])),nominal)
    (OUT/'checks.json').write_text(json.dumps(dict(status='passed',future_labels_excluded=True,train_calibration_months_disjoint=True,
        finite_intervals=True,point_inside_intervals=True,availability='observation_time; release lag unknown'),indent=2))
    print(pd.DataFrame(metrics).pivot(index=['model','point_method'],columns='window',values='mae_rub').round(2).to_string())
    print(pd.DataFrame(aggregate).query('nominal==0.9').round(4).to_string(index=False))


if __name__=='__main__':main()
