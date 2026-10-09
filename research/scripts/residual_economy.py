"""Matured residual correction with equal month weight and fixed source ablations."""
import json
import hashlib
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from backtest import ROOT, load_panel

KEYS=['territory_id','origin','date','horizon']
OUT=ROOT/'results/residual_economy'


def features(history, name, origin, cfg):
    lag=cfg['sensitivity_lag_months'] if name.endswith('lag3') else cfg['rates_lag_months']
    survey_lag=cfg['sensitivity_lag_months'] if name.endswith('lag3') else cfg['survey_lag_months']
    columns=[];values=[]
    for label,delay in [('loan_rate',lag),('deposit_rate',lag),('confidence',survey_lag),('save_preference',survey_lag)]:
        if name=='residual_only':continue
        if name=='residual_credit' and label!='loan_rate':continue
        if name=='residual_savings' and label!='deposit_rate':continue
        if name=='residual_expectations' and label not in ['confidence','save_preference']:continue
        date=origin-pd.DateOffset(months=delay)
        for transform in ['level','change3']:
            value=history.loc[date,label]
            if transform=='change3':value-=history.loc[date-pd.DateOffset(months=3),label]
            columns.append(label+'_'+transform);values.append(value)
    return np.asarray(values,dtype=float),columns


def run_prediction(anchor, history, name, origin, h, cfg, exposures):
    past=anchor[(anchor.horizon==h)&(anchor.date<=origin)&(anchor.origin<origin)]
    training=[];ys=[];weights=[];months=[]
    hierarchical=name=='residual_all_exposures'
    for old,g in past.groupby('origin'):
        f,labels=features(history,name,old,cfg)
        if not np.isfinite(f).all():continue
        err=np.log(g.actual.to_numpy()/g.prediction.to_numpy())
        if hierarchical:
            e=exposures(old,g.territory_id.to_numpy())
            # Six groups, defined only by expenditure and marketplace exposure at old origin.
            group=np.minimum(np.floor((e[:,0]+1)*1.5),2).astype(int)*2+(e[:,1]>0).astype(int)
            present=np.unique(group)
            for key in present:
                mask=group==key;z=e[mask].mean(axis=0)
                training.append(np.r_[f,z,np.outer(f,z).ravel()]);ys.append(np.median(err[mask]));weights.append(1/len(present))
        else:
            training.append(f);ys.append(np.median(err));weights.append(1.)
        months.append(old)
    n=len(months)
    current,labels=features(history,name,origin,cfg)
    ids=anchor[anchor.horizon==h].territory_id.unique()
    ids=np.sort(ids)
    if n<cfg['minimum_training_months']:
        return ids,np.zeros(len(ids)),dict(months=n,active=False),None
    assert np.isfinite(current).all()
    x=np.asarray(training);y=np.asarray(ys);w=np.asarray(weights)
    if hierarchical:
        e=exposures(origin,ids);xf=np.c_[np.tile(current,(len(ids),1)),e,
                                        np.einsum('i,nj->nij',current,e).reshape(len(ids),-1)]
        labels=labels+['exposure_level','exposure_marketplace']+[l+' x '+e for l in labels for e in ['level','marketplace']]
    else:xf=np.tile(current,(len(ids),1))
    if not len(labels):
        raw=np.full(len(ids),np.average(y,weights=w));record={'constant':float(raw[0])}
    else:
        # Scale fitted on matured training inputs only, not on future observation months.
        scaler=StandardScaler().fit(x,sample_weight=w)
        model=Ridge(alpha=cfg['ridge_alpha']).fit(scaler.transform(x),y,sample_weight=w)
        raw=model.predict(scaler.transform(xf));record=dict(model=model,scaler=scaler,features=labels,forecast_features=xf)
    correction=cfg['shrinkage']*np.clip(raw,-cfg['clip_log'],cfg['clip_log'])
    details=dict(months=n,active=True,last_training_target=str(past.date.max().date()),
                 mean_log_correction=float(correction.mean()),min_log_correction=float(correction.min()),
                 max_log_correction=float(correction.max()),features=labels)
    if len(labels):
        details.update(intercept=float(model.intercept_),coefficients=model.coef_.tolist(),
                       scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist())
    else:details['constant']=record['constant']
    return ids,correction,details,record


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/residual_economy.json').read_text())
    source=ROOT/'results/combined_sources/predictions.parquet'
    anchor=pd.read_parquet(source);anchor=anchor[anchor.model==cfg['comparator']].sort_values(KEYS).reset_index(drop=True)
    (OUT/'protocol.json').write_text(json.dumps(dict(config=cfg,started_at=datetime.now(timezone.utc).isoformat(),
        reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256((ROOT/'data/external/financial/manifest.json').read_bytes()).hexdigest(),
        keys=KEYS,training='Only same-horizon baseline errors with target month <= origin. One unit of weight per matured origin month.',
        strict_real_time_eligible=False),indent=2))
    history=pd.read_csv(ROOT/'data/external/financial/history.csv',index_col='date',parse_dates=True)
    ids,dates,cats,values=load_panel();total=cats.index('Все категории');market=cats.index('Маркетплейсы')
    positions=pd.Index(ids)
    def exposures(origin,selected):
        i=dates.get_loc(origin);level=np.log(values[:,i,total]);share=values[:,i,market]/values[:,i,total]
        ranks=pd.Series(level).rank(pct=True).to_numpy()*2-1
        market_rank=pd.Series(share).rank(pct=True).to_numpy()*2-1
        return np.c_[ranks,market_rank][positions.get_indexer(selected)]
    frames=[anchor.copy()];details=[]
    for name in cfg['variants']:
        g=anchor.copy()
        for (origin,h),fold in g.groupby(['origin','horizon']):
            selected,c,info,_=run_prediction(anchor,history,name,origin,h,cfg,exposures)
            mapped=pd.Series(c,index=selected).reindex(fold.territory_id).to_numpy()
            g.loc[fold.index,'prediction']=fold.prediction.to_numpy()*np.exp(mapped)
            details.append(dict(model=name,origin=str(origin.date()),horizon=int(h),**info))
        g['model']=name;frames.append(g);print('Residual candidate completed:',name,flush=True)
    result=pd.concat(frames,ignore_index=True)
    assert np.isfinite(result.prediction).all() and (result.prediction>0).all()
    result.to_parquet(OUT/'predictions.parquet',index=False)
    metrics=[]
    for name,g in result.groupby('model'):
        for window,mask in [('public_origins_jun_sep',g.origin.between('2024-06-01','2024-09-01')),
                            ('later_origins',g.origin>pd.Timestamp('2024-09-01'))]:
            d=g[mask];metrics.append(dict(model=name,window=window,n=len(d),mae_rub=float(abs(d.prediction-d.actual).mean()),
                                        bias_rub=float((d.prediction-d.actual).mean())))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'coefficients.json').write_text(json.dumps(details,indent=2))
    # Corrupt future labels AND source months: correction at this origin must remain identical.
    origin=pd.Timestamp('2024-08-01');bad=anchor.copy();bad.loc[bad.date>origin,'actual']*=1000
    bad_history=history.copy();bad_history.loc[bad_history.index>=origin,:]=9999
    for name in cfg['variants']:
        for h in cfg['horizons']:
            _,a,_,_=run_prediction(anchor,history,name,origin,h,cfg,exposures)
            _,b,_,_=run_prediction(bad,bad_history,name,origin,h,cfg,exposures)
            assert np.array_equal(a,b)
    # Preserve all latest fits; forecasts are conditional on cached foundation model outputs.
    future=pd.read_parquet(ROOT/'results/combined_sources/future_forecasts.parquet')
    future=future[future.model==cfg['comparator']+'_ema'].copy();fits={};future_frames=[];contributions=[];explanations=[]
    for name in cfg['variants']:
        fit_by_h={};g=future.copy()
        for h,fold in g.groupby('horizon'):
            selected,c,info,record=run_prediction(anchor,history,name,dates[-1],h,cfg,exposures)
            g.loc[fold.index,'prediction']=fold.prediction.to_numpy()*np.exp(pd.Series(c,index=selected).reindex(fold.territory_id).to_numpy())
            fit_by_h[int(h)]=dict(fit=record,info=info,ids=selected,log_correction=c)
            if record and 'model' in record:
                parts=record['scaler'].transform(record['forecast_features'])*record['model'].coef_
                raw=parts.sum(axis=1)+record['model'].intercept_
                for j,label in enumerate(record['features']):
                    contributions.append(dict(model=name,horizon=int(h),feature=label,
                        mean_raw_log_contribution=float(parts[:,j].mean()),mean_abs_raw_log_contribution=float(abs(parts[:,j]).mean())))
            else:raw=np.full(len(selected),record['constant'] if record else 0.)
            assert np.allclose(c,cfg['shrinkage']*np.clip(raw,-cfg['clip_log'],cfg['clip_log']),atol=1e-12)
            explanations.append(pd.DataFrame(dict(territory_id=selected,model=name,horizon=h,origin=dates[-1],
                raw_log_correction=raw,clipping_adjustment=np.clip(raw,-cfg['clip_log'],cfg['clip_log'])-raw,
                shrunk_log_correction=c,forecast_multiplier=np.exp(c))))
        g['model']=name;future_frames.append(g);fits[name]=fit_by_h
    joblib.dump(dict(config=cfg,origin=dates[-1],fits=fits,source_history=history),OUT/'trained_corrections.joblib')
    restored=joblib.load(OUT/'trained_corrections.joblib')
    for name in fits:
        for h in fits[name]:
            original=fits[name][h]['fit'];loaded=restored['fits'][name][h]['fit']
            if original and 'model' in original:
                assert np.array_equal(original['model'].predict(original['scaler'].transform(original['forecast_features'])),
                                      loaded['model'].predict(loaded['scaler'].transform(loaded['forecast_features'])))
            assert np.array_equal(fits[name][h]['log_correction'],restored['fits'][name][h]['log_correction'])
    pd.concat(future_frames,ignore_index=True).to_parquet(OUT/'future_forecasts.parquet',index=False)
    pd.concat(explanations,ignore_index=True).to_parquet(OUT/'future_explanations.parquet',index=False)
    pd.DataFrame(contributions).to_csv(OUT/'future_feature_contributions.csv',index=False)
    (OUT/'checks.json').write_text(json.dumps(dict(status='passed',future_labels_excluded=True,future_source_months_excluded=True,
        finite_positive_predictions=True,training_months_not_municipal_row_count=True,joblib_roundtrip=True,
        all_candidates_preserved=True,strict_real_time_eligible=False,future_accuracy='unknown; no 2025 target',
        exposure_scope='past expenditure and marketplace share; demographic data not available'),indent=2))
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=='__main__': main()
