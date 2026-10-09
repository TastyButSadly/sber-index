"""Ablations and paired month-cluster uncertainty for the structural extension."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from panel_features import dataset
from architecture_extensions import structural_features,KEYS
from develop_point_ensemble import factor_forecast


def main():
    out=ROOT/'results/architecture_extensions';cfg=json.loads((ROOT/'configs/architecture_extensions.json').read_text())
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    log=np.log(values);factor=np.median(log,axis=0);residual=log-factor[None,:,:]
    allp=pd.read_parquet(out/'predictions.parquet');base=allp[allp.model=='anchor'].set_index(KEYS).sort_index().reset_index()
    variants={'shares_without_remainder':[j for j in range(18) if j%6!=5],
              'remainder_only':[5,11,17],'share_changes_only':list(range(6,12))}
    prediction={n:base.prediction.to_numpy().copy() for n in variants}
    for (o,h),g in base.groupby(['origin','horizon']):
        i=dates.get_loc(o);pos=g.index.to_numpy()
        x,y,scale,xf,starts=dataset(values[:,:i+1],residual[:,:i+1],i,h,target,np.arange(len(cats)))
        w=scale**cfg['sample_weight_power'];w/=w.mean();ff=factor_forecast(factor[:,target],i,h,'ema')
        xx=np.concatenate([structural_features(values[:,:s+1],residual[:,:s+1],s,h,target) for s in starts])*cfg['extra_feature_scale']
        xt=structural_features(values[:,:i+1],residual[:,:i+1],i,h,target)*cfg['extra_feature_scale']
        with threadpool_limits(2):
            old=np.exp(Ridge(alpha=cfg['ridge_alpha']).fit(x,y,sample_weight=w).predict(xf)+ff)
            for name,columns in variants.items():
                fitted=Ridge(alpha=cfg['ridge_alpha']).fit(np.c_[x,xx[:,columns]],y,sample_weight=w)
                prediction[name][pos]=base.prediction.to_numpy()[pos]+(np.exp(fitted.predict(np.c_[xf,xt[:,columns]])+ff)-old)/3
    frames=[]
    for name,p in prediction.items():
        frame=base.copy();frame['model']=name;frame['prediction']=p;frames.append(frame)
    ablated=pd.concat(frames,ignore_index=True);ablated.to_parquet(out/'ablation_predictions.parquet',index=False)
    combined=pd.concat([allp,ablated],ignore_index=True)
    metrics=[];monthly=[];bootstrap=[]
    for name,g in combined.groupby('model'):
        g=g.set_index(KEYS).reindex(pd.MultiIndex.from_frame(base[KEYS])).reset_index()
        loss=np.abs(g.actual-g.prediction).to_numpy();difference=loss-np.abs(base.actual-base.prediction).to_numpy()
        for window,mask in [('public_origins_jun_sep',g.origin.between('2024-06-01','2024-09-01')),
                            ('later_origins',g.origin>pd.Timestamp('2024-09-01'))]:
            sub=g[mask];metrics.append(dict(model=name,window=window,n=len(sub),mae_rub=float(loss[mask].mean()),
                delta_vs_anchor_rub=float(difference[mask].mean())))
            groups=sub.assign(loss=loss[mask],delta=difference[mask]).groupby('date').agg(
                loss_sum=('loss','sum'),delta_sum=('delta','sum'),n=('loss','size'))
            for date,row in groups.iterrows():monthly.append(dict(model=name,window=window,date=str(date.date()),
                n=int(row.n),mae_rub=row.loss_sum/row.n,delta_vs_anchor_rub=row.delta_sum/row.n))
            if window=='public_origins_jun_sep':
                rng=np.random.default_rng(431);ix=rng.integers(0,len(groups),(10000,len(groups)))
                sums=groups.delta_sum.to_numpy();counts=groups.n.to_numpy()
                draws=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
                lo,hi=np.quantile(draws,[.025,.975])
                bootstrap.append(dict(model=name,target_month_clusters=len(groups),paired_delta_low=lo,paired_delta_high=hi,
                    warning='descriptive; six dependent calendar months; already inspected data; no selection correction'))
    pd.DataFrame(metrics).to_csv(out/'ablation_metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(out/'monthly_stability.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(out/'paired_month_bootstrap.csv',index=False)
    print(pd.DataFrame(metrics).pivot(index='model',columns='window',values='mae_rub').round(2).to_string(),flush=True)


if __name__=='__main__':main()
