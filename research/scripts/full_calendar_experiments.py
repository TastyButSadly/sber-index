"""Federal production-calendar ablations, preserving cached foundation components."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import hashlib
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from panel_features import dataset
from architecture_extensions import structural_features,KEYS
from develop_point_ensemble import factor_forecast

OUT=ROOT/'results/full_calendar'
CAL=ROOT/'data/external/calendar'
OFF={2022:['03-07','05-02','05-03','05-10','06-13'],
     2023:['02-24','05-08','11-06'],
     2024:['04-29','04-30','05-10','12-30','12-31'],
     2025:['05-02','05-08','06-13','11-03','12-31']}
WORK={2022:['03-05'],2023:[],2024:['04-27','11-02','12-28'],2025:['11-01']}
KNOWN={2022:'2021-09-16',2023:'2022-08-29',2024:'2023-08-10',2025:'2024-10-04'}
SOURCE='https://www.cbr.ru/other/holidays/'


def calendar_table():
    days=pd.date_range('2022-01-01','2025-12-31')
    holidays=['01-'+str(i).zfill(2) for i in range(1,9)]+['02-23','03-08','05-01','05-09','06-12','11-04']
    rows=[]
    for d in days:
        md=d.strftime('%m-%d');holiday=md in holidays
        off=d.weekday()>=5 or holiday or md in OFF[d.year]
        if md in WORK[d.year]:off=False
        rows.append(dict(date=d,working=not off,holiday=holiday,working_saturday=(not off and d.weekday()==5)))
    daily=pd.DataFrame(rows);monthly=[]
    for month,g in daily.groupby(daily.date.dt.to_period('M')):
        a=(~g.working).to_numpy();run=best=0
        for value in a:run=run+1 if value else 0;best=max(best,run)
        monthly.append(dict(date=month.to_timestamp(),days=len(g),workdays=int(g.working.sum()),
            offdays=int(a.sum()),working_saturdays=int(g.working_saturday.sum()),
            longest_off_run=best,last7_off=int(a[-7:].sum())))
    return daily,pd.DataFrame(monthly).set_index('date')


def contrast(table,origin,h):
    target=origin+pd.DateOffset(months=h);prior=target-pd.DateOffset(years=1)
    assert pd.Timestamp(KNOWN[target.year])<=origin
    a,b=table.loc[target],table.loc[prior]
    return np.array([np.log(a.days/b.days),np.log(a.workdays/b.workdays),
        a.offdays/a.days-b.offdays/b.days,(a.working_saturdays-b.working_saturdays)/31,
        (a.longest_off_run-b.longest_off_run)/31,(a.last7_off-b.last7_off)/7])


def factor_correction(factor,dates,o,h,table,cfg):
    starts=list(range(12,o-h+1))
    if len(starts)<cfg['factor_min_matured_months']:return 0.,len(starts)
    x=np.array([contrast(table,dates[s],h) for s in starts])
    y=np.array([factor[s+h]-factor_forecast(factor,s,h,'ema') for s in starts])
    # One historical calendar observation per month, rather than copies per municipality.
    fitted=Ridge(alpha=cfg['factor_alpha'],fit_intercept=False).fit(x,y)
    correction=cfg['factor_shrinkage']*np.clip(fitted.predict(contrast(table,dates[o],h)[None])[0],
        -cfg['factor_clip_log'],cfg['factor_clip_log'])
    return float(correction),len(starts)


def main():
    OUT.mkdir(parents=True,exist_ok=True);CAL.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/full_calendar.json').read_text())
    (OUT/'protocol.json').write_text(json.dumps(dict(config=cfg,started_at=datetime.now(timezone.utc).isoformat(),
        features=['log_days_yoy','log_workdays_yoy','off_fraction_yoy','working_saturday_yoy',
                  'longest_off_run_yoy','last7_off_yoy'],source=SOURCE,annual_decree_dates=KNOWN),indent=2))
    daily,table=calendar_table();daily.to_parquet(CAL/'daily.parquet',index=False);table.to_csv(CAL/'monthly.csv')
    totals=daily.groupby(daily.date.dt.year).working.sum().to_dict()
    assert totals=={2022:247,2023:247,2024:248,2025:247},totals
    (CAL/'manifest.json').write_text(json.dumps(dict(source=SOURCE,annual_decree_dates=KNOWN,
        scope=cfg['calendar_scope'],annual_workdays=totals,overrides_off=OFF,overrides_work=WORK,
        note='Federal statutory holidays plus transfers/weekend compensations. Does not describe bank settlement or regional holidays.',
        daily_sha256=hashlib.sha256((CAL/'daily.parquet').read_bytes()).hexdigest()),indent=2))
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    logs=np.log(values);factor=np.median(logs,axis=0);residual=logs-factor[None,:,:]
    macro=pd.read_parquet(ROOT/'data/external/macro/monthly_features.parquet').reindex(dates).to_numpy()
    source=pd.read_parquet(ROOT/'results/combined_sources/predictions.parquet')
    anchor=source[source.model==cfg['comparator']].sort_values(KEYS).reset_index(drop=True)
    library={cfg['comparator']:anchor.prediction.to_numpy().copy()}
    library.update({name:anchor.prediction.to_numpy().copy() for name in cfg['variants']})
    shares_columns=[cats.index('Маркетплейсы'),cats.index('Продовольствие')]
    def extra(s,h):
        structural=structural_features(values[:,:s+1],residual[:,:s+1],s,h,target)*cfg['structure_scale']
        mf=np.tile(macro[s],(len(ids),1));shares=values[:,s,shares_columns]/values[:,s,target,None]
        shares-=np.median(shares,axis=0)
        return np.c_[structural,np.c_[mf,mf*shares[:,0,None],mf*shares[:,1,None]]*cfg['macro_scale']]
    def features(s,h,interactions):
        cal=np.tile(contrast(table,dates[s],h),(len(ids),1))
        if interactions:
            shares=values[:,s,shares_columns]/values[:,s,target,None];shares-=np.median(shares,axis=0)
            cal=np.c_[cal,cal*shares[:,0,None],cal*shares[:,1,None]]
        return cal*cfg['calendar_scale']
    corrections=[];checks=[]
    for (origin,h),g in anchor.groupby(['origin','horizon']):
        o=dates.get_loc(origin);pos=g.index.to_numpy()
        x,y,scale,xf,starts=dataset(values[:,:o+1],residual[:,:o+1],o,h,target,np.arange(len(cats)))
        x=np.c_[x,np.concatenate([extra(s,h) for s in starts])];xf=np.c_[xf,extra(o,h)]
        w=scale**cfg['sample_weight_power'];w/=w.mean();ff=factor_forecast(factor[:,target],o,h,'ema')
        with threadpool_limits(2):
            fitted=Ridge(alpha=cfg['ridge_alpha']).fit(x,y,sample_weight=w);old=np.exp(fitted.predict(xf)+ff)
            for name,interactions in [('calendar_local',False),('calendar_interactions',True)]:
                xc=np.concatenate([features(s,h,interactions) for s in starts]);xfc=features(o,h,interactions)
                model=Ridge(alpha=cfg['ridge_alpha']).fit(np.c_[x,xc],y,sample_weight=w)
                library[name][pos]=anchor.prediction.to_numpy()[pos]+(np.exp(model.predict(np.c_[xf,xfc])+ff)-old)/3
        correction,n=factor_correction(factor[:,target],dates,o,h,table,cfg)
        library['calendar_factor'][pos]=library[cfg['comparator']][pos]*np.exp(correction)
        library['calendar_local_factor'][pos]=library['calendar_local'][pos]*np.exp(correction)
        corrections.append(dict(origin=str(origin.date()),horizon=int(h),matured_training_months=n,
            last_matured_label=str(dates[o].date()),correction_log=correction))
        checks.append(bool(np.isfinite(x).all() and np.isfinite(xf).all()))
        print('Calendar',str(origin.date()),h,flush=True)
    frames=[];metrics=[]
    for name,forecast in library.items():
        assert np.isfinite(forecast).all() and (forecast>0).all()
        frame=anchor.copy();frame['model']=name;frame['prediction']=forecast;frames.append(frame)
        for window,mask in [('public_origins_jun_sep',frame.origin.between('2024-06-01','2024-09-01')),
                            ('later_origins',frame.origin>pd.Timestamp('2024-09-01'))]:
            g=frame[mask];metrics.append(dict(model=name,window=window,n=len(g),
                mae_rub=float(abs(g.prediction-g.actual).mean()),bias_rub=float((g.prediction-g.actual).mean())))
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(corrections).to_csv(OUT/'factor_corrections.csv',index=False)
    o=dates.get_loc('2024-08-01');corrupted=factor[:,target].copy();corrupted[o+1:]+=100
    for h in cfg['horizons']:
        assert factor_correction(factor[:,target],dates,o,h,table,cfg)==factor_correction(corrupted,dates,o,h,table,cfg)
    (OUT/'checks.json').write_text(json.dumps(dict(status='passed',annual_calendar_totals=totals,
        all_features_finite=all(checks),future_factor_labels_excluded=True,calendar_decrees_known_at_origins=True,
        foundation_components_fixed=True,comparison_is_observed_prefix_not_real_time=True),indent=2))
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=='__main__':main()
