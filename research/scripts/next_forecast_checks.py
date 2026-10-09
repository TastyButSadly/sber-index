"""Calendar, shared factor, official policy-rate and matured-label MAE stacking ablations."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from panel_features import dataset
from develop_point_ensemble import factor_forecast

KEYS=['territory_id','origin','date','horizon']
OUT=ROOT/'results/next_forecast_checks'


def calendar(dates,s,h):
    month=dates[s]+pd.DateOffset(months=h);old=month-pd.DateOffset(years=1)
    def weekdays(m):return np.busday_count(m.date(),(m+pd.offsets.MonthBegin(1)).date())
    return np.array([np.log(month.days_in_month/old.days_in_month),np.log(weekdays(month)/weekdays(old))])


def factor_correct(factor,dates,o,h,rates,with_rate=False):
    g=factor[12:o+1]-factor[:o-11]
    base=factor_forecast(factor,o,h,'ema')
    def x(s):
        history=factor[12:s+1]-factor[:s-11]
        ema=np.average(history,weights=.5**np.arange(len(history)-1,-1,-1))
        state=[history[-1]-ema,history[-1]-history[-2] if len(history)>1 else 0,*calendar(dates,s,h)]
        if with_rate:state.extend([(rates[s]-16)/100,(rates[s]-rates[max(11,s-3)])/100])
        return state
    starts=list(range(13,o-h+1))
    starts=[s for s in starts if not with_rate or (np.isfinite(rates[s]) and np.isfinite(rates[max(11,s-3)]))]
    if len(starts)<4:return base
    xx=np.asarray([x(s) for s in starts])
    errors=np.asarray([factor[s+h]-factor_forecast(factor,s,h,'ema') for s in starts])
    model=Ridge(alpha=1).fit(xx,errors)
    return base+.5*np.clip(model.predict(np.asarray([x(o)]))[0],-.05,.05)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/next_round.json').read_text())
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    logs=np.log(values);factors=np.median(logs,axis=0);residual=logs-factors[None,:,:];factor=factors[:,target]
    source=pd.read_parquet(ROOT/'results/reference_audit/predictions.parquet')
    def get(n):return source[source.model==n].set_index(KEYS).sort_index()
    source_new=pd.read_parquet(ROOT/'results/point_ensemble/all_candidate_predictions.parquet')
    base=source_new[source_new.model==cfg['anchor_model']].set_index(KEYS).sort_index()
    d=base[['actual','seasonal_value']].reset_index();idx=base.index
    components=np.empty((len(d),3))
    for k,(name,method) in enumerate([('public_bolt_base','mean1'),('weighted_panel_ridge_public_factor_mean1','mean1'),('public_snaive_growth','mean3')]):
        p=get(name).prediction.reindex(idx).to_numpy()
        for (o,h),g in d.groupby(['origin','horizon']):
            pos=g.index.to_numpy();i=dates.get_loc(o)
            components[pos,k]=p[pos]*np.exp(factor_forecast(factor,i,h,'ema')-factor_forecast(factor,i,h,method))
    anchor=components.mean(axis=1)
    assert np.allclose(anchor,base.prediction,rtol=1e-10)
    library={'anchor_ema':anchor.copy(),
             'calendar_days_half':anchor.copy(),'calendar_days_full':anchor.copy(),
             'calendar_weekdays_quarter':anchor.copy(),'local_calendar_ridge':anchor.copy(),
             'factor_ar_calendar':anchor.copy(),'factor_ar_calendar_rate':anchor.copy(),
             'past_mae_stacking':anchor.copy()}
    ledger=pd.read_parquet(ROOT/'data/external/news/monthly_asof.parquet').set_index('origin')
    rates=ledger.announced_policy_rate.reindex(dates).to_numpy(dtype=float)
    protocol=[]
    for (o,h),g in d.groupby(['origin','horizon']):
        pos=g.index.to_numpy();i=dates.get_loc(o);cal=calendar(dates,i,h)
        library['calendar_days_half'][pos]=anchor[pos]*np.exp(.5*cal[0])
        library['calendar_days_full'][pos]=anchor[pos]*np.exp(cal[0])
        library['calendar_weekdays_quarter'][pos]=anchor[pos]*np.exp(.25*cal[1])
        observed=values[:,:i+1];rr=residual[:,:i+1]
        ds=dataset(observed,rr,i,h,target,np.arange(len(cats)))
        x,y,scale,xf,starts=ds
        xc=np.repeat(np.asarray([calendar(dates,s,h) for s in starts]),len(ids),axis=0)*cfg['calendar_ridge_feature_scale']
        xfc=np.tile(cal,(len(ids),1))*cfg['calendar_ridge_feature_scale']
        w=scale**1.5;w/=w.mean()
        with threadpool_limits(2):model=Ridge(alpha=3).fit(np.c_[x,xc],y,sample_weight=w)
        rfc=np.exp(model.predict(np.c_[xf,xfc])+factor_forecast(factor,i,h,'ema'))
        library['local_calendar_ridge'][pos]=(components[pos,0]+rfc+components[pos,2])/3
        for name,rate in [('factor_ar_calendar',False),('factor_ar_calendar_rate',True)]:
            ff=factor_correct(factor,dates,i,h,rates,rate)
            library[name][pos]=anchor[pos]*np.exp(ff-factor_forecast(factor,i,h,'ema'))
        past=d[(d.date<=o)&(d.origin<o)&(d.horizon==h)]
        months=sorted(past.date.unique())
        if len(months)<3:continue
        past=past[past.date.isin(months[-cfg['stacking_lookback_target_months']:])]
        c=components[past.index];actual=past.actual.to_numpy();scale_mae=np.abs(c.mean(axis=1)-actual).mean()
        def objective(weights):
            return np.mean(np.abs(actual-c@weights))+cfg['stacking_prior_penalty']*scale_mae*np.sum((weights-1/3)**2)
        fit=minimize(objective,np.full(3,1/3),method='SLSQP',bounds=[(0,1)]*3,
                     constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':100,'ftol':1e-7})
        if not fit.success:raise RuntimeError('Stacking optimizer failed: '+fit.message)
        library['past_mae_stacking'][pos]=components[pos]@fit.x
        protocol.append(dict(origin=str(o.date()),horizon=int(h),last_label=str(past.date.max().date()),
            n=len(past),weights=fit.x.tolist(),objective=float(fit.fun)))
    outputs=[];metrics=[];early={}
    for name,p in library.items():
        assert np.isfinite(p).all() and (p>0).all()
        frame=d.copy();frame['model']=name;frame['prediction']=p;outputs.append(frame)
        for window,mask in [('public_origins_jun_sep',frame.origin.between('2024-06-01','2024-09-01')),
                            ('later_origins',frame.origin>pd.Timestamp('2024-09-01')),
                            ('targets_oct_nov',frame.date.between('2024-10-01','2024-11-01'))]:
            sub=frame[mask];metrics.append(dict(model=name,window=window,n=len(sub),mae_rub=float(np.abs(sub.actual-sub.prediction).mean())))
        mask=frame.date<=pd.Timestamp(cfg['calibration_target_cutoff'])
        early[name]=float(np.abs(frame.loc[mask,'actual']-frame.loc[mask,'prediction']).mean())
    pd.concat(outputs,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'weights_protocol.json').write_text(json.dumps(protocol,indent=2))
    (OUT/'early_selection.json').write_text(json.dumps(dict(chosen=min(early,key=early.get),losses=early),indent=2))
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    # Future factor values and future announcements cannot enter the calculation.
    i=dates.get_loc('2024-08-01');bad=factor.copy();bad[i+1:]+=10
    bad_rates=rates.copy();bad_rates[i+1:]=99
    for h in [1,2,3]:
        assert factor_correct(factor,dates,i,h,rates,True)==factor_correct(bad,dates,i,h,bad_rates,True)
    assert all(pd.Timestamp(row['last_label'])<=pd.Timestamp(row['origin']) for row in protocol)
    (OUT/'checks.json').write_text(json.dumps(dict(status='passed',future_factor_and_rates_excluded=True,
        weights_use_matured_labels=True,anchor_reproduced=True,calendar='calendar days and Monday-Friday counts; not Russian production calendar'),indent=2))
    print(pd.DataFrame(metrics).pivot(index='model',columns='window',values='mae_rub').round(2).to_string())
    print('Early choice:',min(early,key=early.get),flush=True)


if __name__=='__main__':main()
