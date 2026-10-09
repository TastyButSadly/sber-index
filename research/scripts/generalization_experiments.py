"""Geographic transfer, temporal purge and algebraically consistent level augmentation."""
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


def main():
    out=ROOT/'results/generalization';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/generalization.json').read_text())
    (out/'protocol.json').write_text(json.dumps(dict(config=cfg,started_at=datetime.now(timezone.utc).isoformat(),
        independent_test=False,synthetic_samples_increase_evaluation_sample_size=False),indent=2))
    ids,dates,cats,values=load_panel();target=cats.index('Все категории');logs=np.log(values)
    common=np.median(logs,axis=0);residual=logs-common[None,:,:]
    source=pd.read_parquet(ROOT/'results/combined_sources/predictions.parquet')
    anchor=source[(source.model=='anchor')&(source.origin>=pd.Timestamp(cfg['evaluation_origins_start']))].set_index(KEYS).sort_index().reset_index()
    paired_index=pd.MultiIndex.from_frame(anchor[KEYS]);pooled=source[source.model=='structure_macro'].set_index(KEYS).reindex(paired_index).prediction.to_numpy()
    names=['anchor','pooled_structure_macro','pooled_augmented','pooled_purged','region_holdout','region_holdout_augmented']
    library={n:anchor.prediction.to_numpy().copy() for n in names};library['pooled_structure_macro']=pooled
    geo=pd.read_parquet(ROOT/'data/processed/municipalities_2024.parquet').drop_duplicates('territory_id').set_index('territory_id')
    regions=geo.region_code.reindex(ids).to_numpy();fold=regions%cfg['geographic_folds']
    macro=pd.read_parquet(ROOT/'data/external/macro/monthly_features.parquet').reindex(dates).to_numpy()
    share_cols=[cats.index('Маркетплейсы'),cats.index('Продовольствие')]
    def extra(local,s,h,train):
        structural=structural_features(values[:,:s+1],local[:,:s+1],s,h,target)*cfg['structure_scale']
        mf=np.tile(macro[s],(len(ids),1));shares=values[:,s,share_cols]/values[:,s,target,None]
        shares-=np.median(shares[train],axis=0)
        return np.c_[structural,np.c_[mf,mf*shares[:,0,None],mf*shares[:,1,None]]*cfg['national_macro_scale']]
    def fit(x,y,w,augmented,seed):
        if augmented:
            rng=np.random.default_rng(seed);shift=rng.normal(0,cfg['synthetic_log_level_sd'],len(y))
            a=x.copy();b=x.copy()
            # A uniform scaling of all categories preserves shares and monthly changes.
            a[:,:3*len(cats)]+=shift[:,None];b[:,:3*len(cats)]-=shift[:,None]
            x=np.r_[x,a,b];y=np.r_[y,y+shift,y-shift]
            w=np.r_[w*.5,w*.25,w*.25]
        with threadpool_limits(2):return Ridge(alpha=cfg['ridge_alpha']).fit(x,y,sample_weight=w)
    reproductions=[];fold_protocol=[]
    for (origin,h),g in anchor.groupby(['origin','horizon']):
        o=dates.get_loc(origin);pos=g.index.to_numpy();all_ids=np.ones(len(ids),dtype=bool)
        x,y,scale,xf,starts=dataset(values[:,:o+1],residual[:,:o+1],o,h,target,np.arange(len(cats)))
        xx=np.concatenate([extra(residual,s,h,all_ids) for s in starts]);xt=extra(residual,o,h,all_ids)
        full=np.c_[x,xx];test=np.c_[xf,xt];w=scale**cfg['sample_weight_power'];w/=w.mean()
        ff=factor_forecast(common[:,target],o,h,'ema')
        with threadpool_limits(2):old=Ridge(alpha=cfg['ridge_alpha']).fit(x,y,sample_weight=w)
        old_prediction=np.exp(old.predict(xf)+ff)
        other_sum=3*library['anchor'][pos]-old_prediction
        fitted=fit(full,y,w,False,0);reproduced=(other_sum+np.exp(fitted.predict(test)+ff))/3
        error=float(np.max(np.abs(reproduced-pooled[pos])));assert error<.01;reproductions.append(error)
        augmented=fit(full,y,w,True,cfg['synthetic_seed']+o*13+h)
        library['pooled_augmented'][pos]=(other_sum+np.exp(augmented.predict(test)+ff))/3
        keep=np.repeat(np.array(starts)+h<=o-cfg['purge_latest_target_months'],len(ids))
        assert keep.sum()>=2*len(ids)
        purged=fit(full[keep],y[keep],w[keep]/w[keep].mean(),False,0)
        library['pooled_purged'][pos]=(other_sum+np.exp(purged.predict(test)+ff))/3
        for k in range(cfg['geographic_folds']):
            held=fold==k;train=~held;assert not set(regions[train])&set(regions[held])
            train_factor=np.median(logs[train,:o+1],axis=0)
            local=logs[:,:o+1]-train_factor[None,:,:]
            x,y,scale,xf,starts=dataset(values[:,:o+1],local,o,h,target,np.arange(len(cats)))
            xx=np.concatenate([extra(local,s,h,train) for s in starts]);xt=extra(local,o,h,train)
            train_rows=np.tile(train,len(starts));full=np.c_[x,xx][train_rows];test=np.c_[xf,xt][held]
            # Each training municipality's weight uses only training-region medians.
            scale=np.concatenate([values[train,s+h-12,target]/np.median(values[train,s+h-12,target]) for s in starts])
            w=scale**cfg['sample_weight_power'];w/=w.mean();y=y[train_rows]
            ff=factor_forecast(train_factor[:,target],o,h,'ema')
            for name,augment in [('region_holdout',False),('region_holdout_augmented',True)]:
                fitted=fit(full,y,w,augment,cfg['synthetic_seed']+o*97+h*7+k)
                prediction=(other_sum[held]+np.exp(fitted.predict(test)+ff))/3
                library[name][pos[held]]=prediction
            fold_protocol.append(dict(origin=str(pd.Timestamp(origin).date()),horizon=int(h),fold=k,
                held_regions=sorted(map(int,np.unique(regions[held]))),training_regions=sorted(map(int,np.unique(regions[train]))),
                train_label_count=len(y),held_municipalities=int(held.sum()),factor_fit_regions='training only',
                information_mode='held municipalities have known observed past; no held labels used in Ridge fitting'))
        print('Generalization',str(pd.Timestamp(origin).date()),h,flush=True)
    outputs=[];metrics=[]
    for name,p in library.items():
        assert np.isfinite(p).all() and (p>0).all()
        g=anchor.copy();g['model']=name;g['prediction']=p;g['geographic_fold']=g.territory_id.map(dict(zip(ids,fold)));outputs.append(g)
        for window,mask in [('public_origins_jun_sep',g.origin.between('2024-06-01','2024-09-01')),
                            ('later_origins',g.origin>pd.Timestamp('2024-09-01'))]:
            metrics.append(dict(model=name,window=window,n=int(mask.sum()),mae_rub=float(np.abs(g.actual[mask]-p[mask]).mean())))
    all_predictions=pd.concat(outputs,ignore_index=True)
    all_predictions.to_parquet(out/'predictions.parquet',index=False)
    diagnostics=all_predictions[all_predictions.origin.between('2024-06-01','2024-09-01')].copy()
    diagnostics['region_code']=diagnostics.territory_id.map(geo.region_code)
    diagnostics['error']=np.abs(diagnostics.actual-diagnostics.prediction)
    diagnostics.groupby(['model','region_code'],as_index=False).agg(n=('error','size'),mae_rub=('error','mean')).to_csv(out/'regional_metrics.csv',index=False)
    diagnostics.groupby(['model','geographic_fold'],as_index=False).agg(n=('error','size'),mae_rub=('error','mean')).to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    (out/'fold_protocol.json').write_text(json.dumps(fold_protocol,indent=2))
    (out/'checks.json').write_text(json.dumps(dict(status='passed',pooled_reproduction_max_rub=max(reproductions),
        geographic_training_and_test_regions_disjoint=True,held_regions_excluded_from_fitted_common_factor=True,
        local_history_known_at_origin=True,synthetic_labels_derived_only_from_matured_training_labels=True,
        synthetic_samples_counted_as_new_evaluation_evidence=False,purged_target_months=cfg['purge_latest_target_months'],
        limitations=['cached Bolt/seasonal experts unchanged and use known local past','not a cold-start test without local observations',
                    'region fold assignment fixed but inspected history is not independent validation']),indent=2))
    print(pd.DataFrame(metrics).pivot(index='model',columns='window',values='mae_rub').round(2).to_string(),flush=True)


if __name__=='__main__':main()
