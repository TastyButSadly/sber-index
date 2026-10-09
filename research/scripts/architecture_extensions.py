"""Replace only the local trainable component; keep cached Bolt/factor/seasonal fixed."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from panel_features import dataset
from develop_point_ensemble import factor_forecast

OUT=ROOT/'results/architecture_extensions'
KEYS=['territory_id','origin','date','horizon']


def structural_features(values,residual,s,h,target):
    total=values[:,:,target]
    columns=[i for i in range(values.shape[2]) if i!=target]
    rest=total-values[:,:,columns].sum(axis=2)
    if (rest<=0).any():raise ValueError('Nonpositive remainder: cannot form log-ratios')
    parts=np.concatenate([values[:,:,columns],rest[:,:,None]],axis=2)
    ratio=np.log(parts/total[:,:,None])
    return np.c_[ratio[:,s],ratio[:,s]-ratio[:,s-3],ratio[:,s+h-12]]


def momentum_features(residual,s,target):
    r=residual[:,:,target]
    return np.c_[r[:,s]-r[:,s-3],r[:,s]-r[:,s-6],r[:,s-2:s+1].mean(axis=1),
                 r[:,s-5:s+1].mean(axis=1),np.median(np.abs(np.diff(r[:,s-5:s+1],axis=1)),axis=1)]


def regional_context(residual,regions,target):
    r=residual[:,:,target];context=np.zeros_like(r)
    for region in np.unique(regions):
        mask=regions==region;n=mask.sum()
        if n<2:continue
        # Leave-own-out mean; no current or future unobserved municipality target.
        context[mask]=(r[mask].sum(axis=0)[None,:]-r[mask])/(n-1)*n/(n+20)
    return context


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/architecture_extensions.json').read_text())
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    log=np.log(values);factor=np.median(log,axis=0);residual=log-factor[None,:,:]
    geo=pd.read_parquet(ROOT/'data/processed/municipalities_2024.parquet').drop_duplicates('territory_id').set_index('territory_id')
    regions=geo.region_code.reindex(ids).to_numpy();region=regional_context(residual,regions,target)
    macro=pd.read_parquet(ROOT/'data/external/macro/monthly_features.parquet').reindex(dates)
    assert not macro.isna().any().any()
    macro_values=macro.to_numpy()
    source=pd.read_parquet(ROOT/'results/point_ensemble/all_candidate_predictions.parquet',filters=[('model','==','bolt_ridge_seasonal_ema')])
    anchor=source.set_index(KEYS).sort_index();d=anchor[['actual','seasonal_value']].reset_index()
    old=pd.read_parquet(ROOT/'results/reference_audit/predictions.parquet',filters=[('model','==','weighted_panel_ridge_public_factor_mean1')]).set_index(KEYS)
    old_ridge=old.prediction.reindex(anchor.index).to_numpy()
    for (o,h),g in d.groupby(['origin','horizon']):
        i=dates.get_loc(o);pos=g.index.to_numpy()
        old_ridge[pos]*=np.exp(factor_forecast(factor[:,target],i,h,'ema')-factor_forecast(factor[:,target],i,h,'mean1'))
    names=['anchor','momentum','composition','regional','macro_interactions','structure_region','structure_region_macro','huber_local','hist_mae_local']
    library={n:anchor.prediction.to_numpy().copy() for n in names}
    def extra(family,s,h):
        parts=[]
        if family in ['momentum','structure_region','structure_region_macro']:parts.append(momentum_features(residual,s,target)*cfg['extra_feature_scale'])
        if family in ['composition','structure_region','structure_region_macro']:parts.append(structural_features(values[:,:s+1],residual[:,:s+1],s,h,target)*cfg['extra_feature_scale'])
        if family in ['regional','structure_region','structure_region_macro']:
            parts.append(np.c_[region[:,s],region[:,s]-region[:,s-1],region[:,s-2:s+1].mean(axis=1),region[:,s+h-12]]*cfg['extra_feature_scale'])
        if family in ['macro_interactions','structure_region_macro']:
            mf=np.tile(macro_values[s],(len(ids),1));shares=values[:,s,[cats.index('Маркетплейсы'),cats.index('Продовольствие')]]/values[:,s,target,None]
            shares-=np.median(shares,axis=0)
            parts.append(np.c_[mf,mf*shares[:,0,None],mf*shares[:,1,None]]*cfg['macro_feature_scale'])
        return np.concatenate(parts,axis=1)
    checks=[]
    for (o,h),g in d.groupby(['origin','horizon']):
        pos=g.index.to_numpy();i=dates.get_loc(o)
        ds=dataset(values[:,:i+1],residual[:,:i+1],i,h,target,np.arange(len(cats)))
        x,y,scale,xf,starts=ds;w=scale**cfg['sample_weight_power'];w/=w.mean()
        ff=factor_forecast(factor[:,target],i,h,'ema')
        with threadpool_limits(2):
            base=Ridge(alpha=cfg['ridge_alpha']).fit(x,y,sample_weight=w)
            reproduced=np.exp(base.predict(xf)+ff)
            assert np.allclose(reproduced,old_ridge[pos],rtol=1e-8,atol=.01)
            for family in names[1:7]:
                xx=np.concatenate([extra(family,s,h) for s in starts]);xt=extra(family,i,h)
                fit=Ridge(alpha=cfg['ridge_alpha']).fit(np.c_[x,xx],y,sample_weight=w)
                pred=np.exp(fit.predict(np.c_[xf,xt])+ff)
                library[family][pos]=anchor.prediction.to_numpy()[pos]+(pred-old_ridge[pos])/3
            scaler=StandardScaler().fit(x)
            robust=HuberRegressor(alpha=cfg['huber_alpha'],epsilon=cfg['huber_epsilon'],max_iter=500).fit(scaler.transform(x),y,sample_weight=w)
            hist=HistGradientBoostingRegressor(loss='absolute_error',early_stopping=False,random_state=42,**cfg['hist_mae']).fit(x,y,sample_weight=w)
            for name,fit,test in [('huber_local',robust,scaler.transform(xf)),('hist_mae_local',hist,xf)]:
                pred=np.exp(fit.predict(test)+ff)
                library[name][pos]=anchor.prediction.to_numpy()[pos]+(pred-old_ridge[pos])/3
        checks.append(float(np.max(np.abs(reproduced-old_ridge[pos]))))
        print('Finished local extensions',str(pd.Timestamp(o).date()),h,flush=True)
    # A fixed shrinkage selector also uses only already matured forecast errors.
    p=library['anchor'].copy();protocol=[]
    for o in sorted(d.origin.unique()):
        if o<pd.Timestamp('2024-06-01'):continue
        past=d[(d.date<=o)&(d.origin<o)];months=sorted(past.date.unique())
        if len(months)<3:continue
        past=past[past.date.isin(months[-4:])];test=d.origin==o
        losses={n:float(np.abs(past.actual-library[n][past.index]).mean()) for n in names}
        chosen=min(losses,key=losses.get);p[test]=.5*library['anchor'][test]+.5*library[chosen][test]
        protocol.append(dict(origin=str(pd.Timestamp(o).date()),last_target=str(past.date.max().date()),chosen=chosen,losses=losses))
    library['past_selection_half']=p
    output=[];metrics=[];early={}
    for name,p in library.items():
        assert np.isfinite(p).all() and (p>0).all()
        g=d.copy();g['model']=name;g['prediction']=p;output.append(g)
        for window,mask in [('public_origins_jun_sep',g.origin.between('2024-06-01','2024-09-01')),
                            ('later_origins',g.origin>pd.Timestamp('2024-09-01')),
                            ('targets_oct_nov',g.date.between('2024-10-01','2024-11-01'))]:
            sub=g[mask];metrics.append(dict(model=name,window=window,n=len(sub),mae_rub=float(np.abs(sub.actual-sub.prediction).mean())))
        mask=g.date<=pd.Timestamp(cfg['early_target_cutoff']);early[name]=float(np.abs(g.loc[mask,'actual']-g.loc[mask,'prediction']).mean())
    pd.concat(output,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'early_selection.json').write_text(json.dumps(dict(chosen=min(names,key=early.get),losses=early),indent=2))
    (OUT/'selector_protocol.json').write_text(json.dumps(protocol,indent=2))
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    # Features use current/lagged values; corrupting later observations has no effect.
    s=dates.get_loc('2024-08-01');bad=values.copy();bad[:,s+1:]*=1000
    bad_log=np.log(bad);bad_r=bad_log-np.median(bad_log,axis=0)[None,:,:]
    for h in cfg['horizons']:
        assert np.array_equal(structural_features(values,residual,s,h,target),structural_features(bad,bad_r,s,h,target))
        assert np.array_equal(momentum_features(residual,s,target),momentum_features(bad_r,s,target))
    assert np.array_equal(regional_context(residual,regions,target)[:,:s+1],regional_context(bad_r,regions,target)[:,:s+1])
    (OUT/'checks.json').write_text(json.dumps(dict(status='passed',baseline_reproduction_max_rub=max(checks),
        future_observations_excluded=True,selector_uses_matured_labels=True,other_category_nonpositive_fraction=0,
        region_mapping='2024 retrospective mapping; no historical publication vintages',macro_release_dates='assumed conservative lags; not verified'),indent=2))
    print(pd.DataFrame(metrics).pivot(index='model',columns='window',values='mae_rub').round(2).to_string(),flush=True)


if __name__=='__main__':main()
