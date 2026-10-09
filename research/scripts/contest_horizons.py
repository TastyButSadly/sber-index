"""Paired Prophet and ensemble comparison on all four contest horizons."""
import logging
import argparse
import json
import gc
import numpy as np
import pandas as pd
from backtest import ROOT,load_panel
from forecast_point import PointEnsemble

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
OUT=ROOT/'results/contest_horizons'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['ensemble','prophet','combine'],default='combine')
    args=parser.parse_args()
    if args.phase=='prophet':
        from prophet import Prophet
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/contest_comparison.json').read_text())
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    order=np.argsort(values[:,0,target]);positions=order[np.linspace(0,len(ids)-1,cfg['sample_territories']).astype(int)]
    pd.DataFrame({'territory_id':ids[positions],'jan2023_spending':values[positions,0,target]}).to_csv(OUT/'sample.csv',index=False)
    checkpoint=OUT/(args.phase+'_checkpoint.parquet')
    frames=[pd.read_parquet(checkpoint)] if checkpoint.exists() else []
    old=OUT/'forecast_checkpoint.parquet'
    if not frames and old.exists() and args.phase!='combine':
        seed=pd.read_parquet(old)
        mask=seed.model.str.startswith('bolt_') if args.phase=='ensemble' else seed.model.isin(['prophet_monthly','seasonal_naive'])
        frames=[seed.loc[mask].copy()]
    done=set(frames[0].origin.unique()) if frames else set()
    for raw_o in (cfg['origins'] if args.phase!='combine' else []):
        if pd.Timestamp(raw_o) in done:continue
        o=dates.get_loc(raw_o);hs=[h for h in cfg['horizons'] if o+h<len(dates)]
        if args.phase=='prophet':
            prediction=pd.DataFrame(columns=['territory_id','origin','date','horizon','model','prediction'])
        elif o>=14:
            model=PointEnsemble(hs).fit(values[:,:o+1],ids,dates[:o+1],cats)
            model.ids=model.ids[positions];model.bolt_context=model.bolt_context[positions]
            model.ridge={h:(m,x[positions]) for h,(m,x) in model.ridge.items()}
            model.seasonal_local={h:r[positions] for h,r in model.seasonal_local.items()}
            prediction=model.predict()
            prediction=prediction[prediction.model=='bolt_ridge_seasonal_ema']
        else:
            # Early long-horizon context has no matured Ridge training pairs.
            import torch
            from chronos import BaseChronosPipeline
            torch.set_num_threads(2)
            logs=np.log(values[:,:o+1,target]);factor=np.median(logs,axis=0);r=logs-factor[None,:]
            pipe=BaseChronosPipeline.from_pretrained(str(ROOT/'models/chronos-bolt-base'),device_map='cpu',torch_dtype=torch.float32)
            q,_=pipe.predict_quantiles([torch.tensor(row,dtype=torch.float32) for row in r[positions]],prediction_length=max(hs),quantile_levels=[.5])
            arr=q[...,0].numpy();parts=[]
            del pipe
            gc.collect()
            for h in hs:
                sf=values[positions,o+h-12,target]
                bolt=np.exp(arr[:,h-1]+factor[o+h-12])
                parts.append(pd.DataFrame(dict(territory_id=ids[positions],origin=dates[o],date=dates[o+h],horizon=h,
                    model='bolt_seasonal_fallback',prediction=(bolt+sf)/2)))
            prediction=pd.concat(parts,ignore_index=True)
        if len(prediction):frames.append(prediction[['territory_id','origin','date','horizon','model','prediction']])
        rows=[]
        for n,i in enumerate(positions if args.phase=='prophet' else []):
            model=Prophet(yearly_seasonality=cfg['prophet_yearly_fourier_order'],weekly_seasonality=False,daily_seasonality=False,
                seasonality_mode=cfg['prophet_seasonality_mode'],changepoint_prior_scale=cfg['prophet_changepoint_prior_scale'],uncertainty_samples=0)
            model.fit(pd.DataFrame({'ds':dates[:o+1],'y':values[i,:o+1,target]}),seed=cfg['seed'])
            pred=model.predict(pd.DataFrame({'ds':[dates[o+h] for h in hs]})).yhat.to_numpy()
            for h,p in zip(hs,pred):
                rows.append(dict(territory_id=ids[i],origin=dates[o],date=dates[o+h],horizon=h,model='prophet_monthly',prediction=p))
                rows.append(dict(territory_id=ids[i],origin=dates[o],date=dates[o+h],horizon=h,model='seasonal_naive',prediction=values[i,o+h-12,target]))
            if (n+1)%32==0:print('Prophet',raw_o,n+1,'/',len(positions),flush=True)
        if rows:frames.append(pd.DataFrame(rows))
        pd.concat(frames,ignore_index=True).to_parquet(checkpoint,index=False)
    if args.phase!='combine':
        pd.concat(frames,ignore_index=True).to_parquet(checkpoint,index=False)
        print('Completed',args.phase,flush=True)
        return
    frames=[pd.read_parquet(OUT/(phase+'_checkpoint.parquet')) for phase in ['ensemble','prophet']]
    result=pd.concat(frames,ignore_index=True)
    actual=pd.DataFrame([dict(territory_id=ids[i],date=date,actual=values[i,t,target]) for i in positions for t,date in enumerate(dates)])
    result=result.merge(actual,on=['territory_id','date'],validate='many_to_one')
    result.to_parquet(OUT/'predictions.parquet',index=False)
    metrics=[]
    for (name,h),g in result.groupby(['model','horizon']):
        error=g.actual-g.prediction
        metrics.append(dict(model=name,horizon=h,n=len(g),origins=g.origin.nunique(),mae_rub=float(np.abs(error).mean()),
            r2_level=float(1-np.sum(error**2)/np.sum((g.actual-g.actual.mean())**2))))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    # Unified deployed algorithm with explicit early fallback, shared keys with Prophet.
    ensemble=result[~result.model.isin(['prophet_monthly','seasonal_naive'])].set_index(['territory_id','origin','date','horizon'])
    prophet=result[result.model=='prophet_monthly'].set_index(['territory_id','origin','date','horizon'])
    assert ensemble.index.equals(prophet.reindex(ensemble.index).index)
    paired=[]
    for h in cfg['horizons']:
        mask=ensemble.index.get_level_values('horizon')==h;e=ensemble[mask];p=prophet.reindex(e.index)
        assert p.prediction.notna().all()
        assert e.index.is_unique and np.array_equal(e.actual,p.actual)
        paired.append(dict(horizon=h,n=len(e),origins=e.reset_index().origin.nunique(),
            ensemble_mae=float(np.abs(e.actual-e.prediction).mean()),prophet_mae=float(np.abs(p.actual-p.prediction).mean())))
    pd.DataFrame(paired).to_csv(OUT/'paired_metrics.csv',index=False)
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    print(pd.DataFrame(paired).round(2).to_string(index=False),flush=True)


if __name__=='__main__':main()
