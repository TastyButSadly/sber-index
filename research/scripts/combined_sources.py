"""Paired ablations of structure + national macro + new regional sources."""
import hashlib
import json
import joblib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from panel_features import dataset
from architecture_extensions import structural_features,KEYS
from develop_point_ensemble import factor_forecast
from forecast_point import PointEnsemble


def main():
    out=ROOT/'results/combined_sources';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/combined_sources.json').read_text())
    # Save fixed candidate definitions before scoring this round.
    (out/'protocol.json').write_text(json.dumps(dict(config=cfg,run_started_at=datetime.now(timezone.utc).isoformat(),
        sources_manifest_sha256=hashlib.sha256((ROOT/'data/external/regional/manifest.json').read_bytes()).hexdigest()),indent=2))
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    log=np.log(values);factor=np.median(log,axis=0);residual=log-factor[None,:,:]
    previous=pd.read_parquet(ROOT/'results/architecture_extensions/predictions.parquet')
    anchor=previous[previous.model=='anchor'].set_index(KEYS).sort_index().reset_index()
    index=pd.MultiIndex.from_frame(anchor[KEYS]);library={}
    for name in ['anchor','composition','macro_interactions']:
        library[name]=previous[previous.model==name].set_index(KEYS).reindex(index).prediction.to_numpy()
    for name in cfg['variants']:library[name]=library['anchor'].copy()
    macro=pd.read_parquet(ROOT/'data/external/macro/monthly_features.parquet').reindex(dates).to_numpy()
    geo=pd.read_parquet(ROOT/'data/processed/municipalities_2024.parquet').drop_duplicates('territory_id').set_index('territory_id')
    regions=geo.region_code.reindex(ids).to_numpy()
    regional=pd.read_parquet(ROOT/'data/external/regional/features.parquet').set_index(['region_code','date'])
    cols=regional.columns.tolist();r=regional.reindex(pd.MultiIndex.from_product([regions,dates])).to_numpy().reshape(len(ids),len(dates),len(cols))
    price_columns=[j for j,c in enumerate(cols) if c.startswith('cpi_')]
    wage_columns=[j for j,c in enumerate(cols) if c.startswith('wage_')]
    def extra(name,s,h,observed=values,local=residual,rf=r):
        parts=[]
        if name.startswith('structure_'):
            parts.append(structural_features(observed[:,:s+1],local[:,:s+1],s,h,target)*cfg['structure_scale'])
            mf=np.tile(macro[s],(len(ids),1))
            shares=observed[:,s,[cats.index('Маркетплейсы'),cats.index('Продовольствие')]]/observed[:,s,target,None]
            shares-=np.median(shares,axis=0)
            parts.append(np.c_[mf,mf*shares[:,0,None],mf*shares[:,1,None]]*cfg['national_macro_scale'])
        columns=[]
        if 'prices' in name:columns+=price_columns
        if 'wages' in name:columns+=wage_columns
        if columns:parts.append(rf[:,s,columns]*cfg['regional_scale'])
        result=np.concatenate(parts,axis=1)
        assert np.isfinite(result).all()
        return result
    for (o,h),g in anchor.groupby(['origin','horizon']):
        i=dates.get_loc(o);pos=g.index.to_numpy()
        x,y,scale,xf,starts=dataset(values[:,:i+1],residual[:,:i+1],i,h,target,np.arange(len(cats)))
        w=scale**cfg['sample_weight_power'];w/=w.mean();ff=factor_forecast(factor[:,target],i,h,'ema')
        with threadpool_limits(2):
            old=np.exp(Ridge(alpha=cfg['ridge_alpha']).fit(x,y,sample_weight=w).predict(xf)+ff)
            for name in cfg['variants']:
                xx=np.concatenate([extra(name,s,h) for s in starts]);xt=extra(name,i,h)
                fitted=Ridge(alpha=cfg['ridge_alpha']).fit(np.c_[x,xx],y,sample_weight=w)
                library[name][pos]=library['anchor'][pos]+(np.exp(fitted.predict(np.c_[xf,xt])+ff)-old)/3
        print('Combined sources',str(pd.Timestamp(o).date()),h,flush=True)
    library['structure_macro_blend']=.5*library['composition']+.5*library['macro_interactions']
    frames=[];metrics=[]
    for name,p in library.items():
        assert np.isfinite(p).all() and (p>0).all()
        g=anchor.copy();g['model']=name;g['prediction']=p;frames.append(g)
        for window,mask in [('public_origins_jun_sep',g.origin.between('2024-06-01','2024-09-01')),
                            ('later_origins',g.origin>pd.Timestamp('2024-09-01'))]:
            metrics.append(dict(model=name,window=window,n=int(mask.sum()),mae_rub=float(np.abs(g.actual[mask]-p[mask]).mean())))
    pd.concat(frames,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    s=dates.get_loc('2024-08-01');bad=values.copy();bad[:,s+1:]*=1000;bad_r=r.copy();bad_r[:,s+1:]=999
    logs=np.log(bad);bad_local=logs-np.median(logs,axis=0)[None,:,:]
    for name in cfg['variants']:
        for h in cfg['horizons']:assert np.array_equal(extra(name,s,h),extra(name,s,h,bad,bad_local,bad_r))
    (out/'checks.json').write_text(json.dumps(dict(status='passed',future_inputs_excluded=True,
        national_sources_latest_snapshot=True,regional_sources_latest_snapshot=True,strict_real_time_eligible=False,
        lag_months={'cpi':2,'wages':3},region_mapping='2024 retrospective dictionary'),indent=2))
    # Train every fixed extension on the latest observed prefix, retaining matched cached outputs.
    point=PointEnsemble.load(ROOT/'results/point_forecast/trained_model.joblib')
    assert np.array_equal(ids,point.ids) and point.origin==dates[-1]
    cached=pd.read_parquet(ROOT/'results/point_forecast/predictions.parquet')
    i=len(dates)-1;trained={};future=[]
    share_categories=[c for c in cats if c!='Все категории']+['Вычисленный остаток']
    base_labels=[(group,c) for group in ['current','previous_month','seasonal_month','monthly_change'] for c in cats]
    macro_names=pd.read_parquet(ROOT/'data/external/macro/monthly_features.parquet').columns.tolist()
    for name in cfg['variants']:
        trained[name]={};labels=base_labels.copy()
        if name.startswith('structure_'):
            labels += [(group,c) for group in ['share_current','share_change3','share_seasonal'] for c in share_categories]
            labels += [(group,c) for group in ['national_macro','macro_x_marketplace_share','macro_x_food_share'] for c in macro_names]
        if 'prices' in name:labels += [('regional_price',cols[j]) for j in price_columns]
        if 'wages' in name:labels += [('regional_wage',cols[j]) for j in wage_columns]
        for h in cfg['horizons']:
            x,y,scale,xf,starts=dataset(values,residual,i,h,target,np.arange(len(cats)))
            xx=np.concatenate([extra(name,s,h) for s in starts]);xt=extra(name,i,h)
            full=np.c_[x,xx];test=np.c_[xf,xt];w=scale**cfg['sample_weight_power'];w/=w.mean()
            assert len(labels)==full.shape[1]
            with threadpool_limits(2):fit=Ridge(alpha=cfg['ridge_alpha']).fit(full,y,sample_weight=w)
            trained[name][h]=dict(model=fit,forecast_features=test,reference_features=np.average(full,axis=0,weights=w),feature_labels=labels)
            for method in ['mean1','ema']:
                frame=cached[(cached.horizon==h)&(cached.model=='bolt_ridge_seasonal_'+method)].set_index('territory_id').reindex(ids).reset_index()
                ff=factor_forecast(point.factor,i,h,method)
                frame['ridge_prediction']=np.exp(fit.predict(test)+ff)
                frame['prediction']=frame[['bolt_prediction','ridge_prediction','seasonal_prediction']].mean(axis=1)
                frame['model']=name+'_'+method;future.append(frame)
    joblib.dump(dict(origin=point.origin,ids=ids,categories=cats,config=cfg,trained=trained),out/'trained_extensions.joblib')
    restored=joblib.load(out/'trained_extensions.joblib')
    for name in trained:
        for h,record in trained[name].items():
            other=restored['trained'][name][h]
            assert np.array_equal(record['model'].predict(record['forecast_features']),other['model'].predict(other['forecast_features']))
    pd.concat(future,ignore_index=True).to_parquet(out/'future_forecasts.parquet',index=False)
    (out/'future_manifest.json').write_text(json.dumps(dict(origin=str(point.origin.date()),horizons=cfg['horizons'],
        variants=cfg['variants'],trained_on_observed_prefix_only=True,joblib_roundtrip='passed',
        cached_components_source='results/point_forecast',future_accuracy='unknown; no 2025 labels',private_local_only=True),indent=2))
    print(pd.DataFrame(metrics).pivot(index='model',columns='window',values='mae_rub').round(2).to_string(),flush=True)


if __name__=='__main__':main()
