"""Fit a structural Ridge extension and reuse explicitly matched cached components."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from panel_features import dataset
from forecast_point import PointEnsemble
from architecture_extensions import structural_features
from develop_point_ensemble import factor_forecast


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',default='results/point_forecast')
    parser.add_argument('--output',default='results/structural_forecast')
    args=parser.parse_args();cache=ROOT/args.cache;out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    state=PointEnsemble.load(cache/'trained_model.joblib')
    cfg=json.loads((ROOT/'configs/architecture_extensions.json').read_text())
    ids,dates,cats,values=load_panel();o=dates.get_loc(state.origin);values=values[:,:o+1]
    assert np.array_equal(ids,state.ids) and cats==state.categories
    target=cats.index('Все категории');log=np.log(values);factor=np.median(log,axis=0)
    residual=log-factor[None,:,:];assert np.array_equal(residual[:,:,target],state.bolt_context)
    cached=pd.read_parquet(cache/'predictions.parquet');frames=[];trained={}
    for h in [1,2,3]:
        x,y,scale,xf,starts=dataset(values,residual,o,h,target,np.arange(len(cats)))
        xx=np.concatenate([structural_features(values[:,:s+1],residual[:,:s+1],s,h,target) for s in starts])*cfg['extra_feature_scale']
        xt=structural_features(values,residual,o,h,target)*cfg['extra_feature_scale']
        w=scale**cfg['sample_weight_power'];w/=w.mean()
        with threadpool_limits(2):fit=Ridge(alpha=cfg['ridge_alpha']).fit(np.c_[x,xx],y,sample_weight=w)
        trained[h]=dict(model=fit,forecast_features=np.c_[xf,xt])
        for method in ['mean1','ema']:
            frame=cached[(cached.horizon==h)&(cached.model=='bolt_ridge_seasonal_'+method)].set_index('territory_id').reindex(ids).reset_index()
            assert len(frame)==len(ids) and frame.origin.eq(state.origin).all()
            ff=factor_forecast(state.factor,o,h,method)
            old,old_features=state.ridge[h]
            assert np.allclose(np.exp(old.predict(old_features)+ff),frame.ridge_prediction,rtol=1e-12)
            frame['ridge_prediction']=np.exp(fit.predict(np.c_[xf,xt])+ff)
            frame['prediction']=frame[['bolt_prediction','ridge_prediction','seasonal_prediction']].mean(axis=1)
            frame['model']='bolt_structural_ridge_seasonal_'+method
            frames.append(frame)
    p=pd.concat(frames,ignore_index=True);assert np.isfinite(p.prediction).all() and (p.prediction>0).all()
    p.to_parquet(out/'predictions.parquet',index=False);p.to_csv(out/'forecast.csv',index=False,encoding='utf-8-sig')
    joblib.dump(dict(origin=state.origin,ids=ids,categories=cats,config=cfg,trained=trained),out/'trained_extension.joblib')
    reloaded=joblib.load(out/'trained_extension.joblib')
    for h in trained:
        assert np.array_equal(reloaded['trained'][h]['model'].predict(reloaded['trained'][h]['forecast_features']),
                              trained[h]['model'].predict(trained[h]['forecast_features']))
    audit=None
    historical=ROOT/'results/architecture_extensions/predictions.parquet'
    if historical.exists():
        reference=pd.read_parquet(historical,filters=[('model','==','composition'),('origin','==',state.origin)])
        if len(reference):
            keys=['territory_id','origin','date','horizon']
            joined=p[p.model=='bolt_structural_ridge_seasonal_ema'].merge(reference,on=keys,suffixes=('_fresh','_audit'),validate='one_to_one')
            assert len(joined)==len(reference)
            audit=float(np.max(np.abs(joined.prediction_fresh-joined.prediction_audit)))
            assert audit<.01
    (out/'manifest.json').write_text(json.dumps(dict(origin=str(state.origin.date()),horizons=[1,2,3],
        status='research candidate; not independently validated',cache_source=args.cache,
        private_local_only=True,trained_on_observed_prefix_only=True,joblib_roundtrip='passed',
        backtest_reproduction_max_rub=audit,
        structure_features='five category/total ratios and derived residual share: current, 3-month change, seasonal',
        future_accuracy='unknown where labels unavailable'),indent=2))
    print('Saved structural model and forecasts',out,'historical reproduction error',audit,flush=True)


if __name__=='__main__':main()
