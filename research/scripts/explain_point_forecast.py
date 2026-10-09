"""Exact accounting explanations of saved forecasts; no causal or SHAP claim."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from backtest import ROOT,load_panel
from forecast_point import PointEnsemble
from panel_features import dataset
from develop_point_ensemble import factor_forecast
from architecture_extensions import structural_features


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--structural',action='store_true');args=parser.parse_args()
    out=ROOT/('results/structural_interpretability' if args.structural else 'results/interpretability');out.mkdir(parents=True,exist_ok=True)
    state=PointEnsemble.load(ROOT/'results/point_forecast/trained_model.joblib')
    ids,dates,cats,values=load_panel();o=dates.get_loc(state.origin)
    values=values[:,:o+1];logs=np.log(values);factor=np.median(logs,axis=0)
    residual=logs-factor[None,:,:];target=cats.index('Все категории')
    assert np.array_equal(ids,state.ids)
    stem='bolt_structural_ridge_seasonal_' if args.structural else 'bolt_ridge_seasonal_'
    saved=pd.read_parquet(ROOT/('results/structural_forecast/predictions.parquet' if args.structural else 'results/point_forecast/predictions.parquet'))
    extension=joblib.load(ROOT/'results/structural_forecast/trained_extension.joblib') if args.structural else None
    features=[];summary=[];errors=[]
    groups=['current','previous_month','seasonal_month','monthly_change']
    for h in [1,2,3]:
        model,xf=state.ridge[h]
        x,y,scale,reproduced,starts=dataset(values,residual,o,h,target,np.arange(len(cats)))
        assert np.array_equal(xf,reproduced)
        labels=[(group,category) for group in groups for category in cats]
        if extension is not None:
            feature_scale=extension['config']['extra_feature_scale']
            xx=np.concatenate([structural_features(values[:,:s+1],residual[:,:s+1],s,h,target) for s in starts])*feature_scale
            x=np.c_[x,xx];xf=extension['trained'][h]['forecast_features'];model=extension['trained'][h]['model']
            share_cats=[c for c in cats if c!='Все категории']+['Вычисленный остаток']
            labels += [(group,category) for group in ['share_current','share_change3','share_seasonal'] for category in share_cats]
        w=scale**state.weight_power
        reference=np.average(x,axis=0,weights=w)
        base_local=float(model.predict(reference[None,:])[0])
        contributions=(xf-reference)*model.coef_[None,:]
        delta=contributions.sum(axis=1)
        assert np.allclose(base_local+delta,model.predict(xf),atol=1e-12)
        multiplier=np.ones_like(delta)
        np.divide(np.expm1(delta),delta,out=multiplier,where=np.abs(delta)>1e-12)
        for method in ['mean1','ema']:
            frame=saved[(saved.horizon==h)&(saved.model==stem+method)].set_index('territory_id').reindex(ids)
            ff=factor_forecast(state.factor,o,h,method)
            baseline=np.exp(ff+base_local)*np.ones(len(ids))
            rub=baseline[:,None]*multiplier[:,None]*contributions/3
            assert np.allclose(baseline+3*rub.sum(axis=1),frame.ridge_prediction,rtol=1e-12,atol=1e-7)
            bolt=(frame.bolt_prediction.to_numpy()-baseline)/3
            seasonal=(frame.seasonal_prediction.to_numpy()-baseline)/3
            reconstructed=baseline+bolt+seasonal+rub.sum(axis=1)
            errors.append(float(np.max(np.abs(reconstructed-frame.prediction))))
            assert np.allclose(reconstructed,frame.prediction,rtol=1e-12,atol=1e-7)
            item=pd.DataFrame(dict(territory_id=ids,origin=state.origin,date=frame.date.to_numpy(),horizon=h,model=frame.model.to_numpy(),
                prediction=frame.prediction.to_numpy(),reference_prediction=baseline,bolt_contribution_rub=bolt,
                seasonal_contribution_rub=seasonal,ridge_feature_contribution_rub=rub.sum(axis=1)))
            # Hold local forecasts fixed and remove the common YoY factor growth.
            global_growth=ff-state.factor[o+h-12]
            item['common_factor_log_growth']=global_growth
            item['common_factor_effect_rub']=item.prediction*(1-np.exp(-global_growth))
            item['local_log_growth_vs_seasonal']=np.log(item.prediction.to_numpy()/values[:,o+h-12,target])-global_growth
            summary.append(item)
            for j in range(xf.shape[1]):
                features.append(pd.DataFrame(dict(territory_id=ids,horizon=h,model=stem+method,
                    feature_group=labels[j][0],category=labels[j][1],
                    value=xf[:,j],reference_value=reference[j],coefficient=model.coef_[j],
                    local_log_contribution=contributions[:,j],ensemble_contribution_rub=rub[:,j])))
    feature=pd.concat(features,ignore_index=True);summary=pd.concat(summary,ignore_index=True)
    feature.to_parquet(out/'feature_contributions.parquet',index=False)
    summary.to_csv(out/'forecast_decomposition.csv',index=False,encoding='utf-8-sig')
    aggregated=feature.assign(abs_contribution=feature.ensemble_contribution_rub.abs()).groupby(
        ['model','horizon','feature_group','category'],as_index=False).agg(
            mean_abs_contribution_rub=('abs_contribution','mean'),mean_contribution_rub=('ensemble_contribution_rub','mean'))
    aggregated.to_csv(out/'feature_summary.csv',index=False,encoding='utf-8-sig')
    cases=feature[feature.territory_id.isin([1673,1665,1333])]
    cases.to_csv(out/'case_feature_contributions.csv',index=False,encoding='utf-8-sig')
    summary[summary.territory_id.isin([1673,1665,1333])].to_csv(out/'case_forecasts.csv',index=False,encoding='utf-8-sig')
    (out/'checks.json').write_text(json.dumps(dict(status='passed',max_reconstruction_error_rub=max(errors),
        origin=str(state.origin.date()),forecast_horizons=[1,2,3],trained_on_observed_prefix_only=True,
        explanation='weighted training mean reference; straight path in linear log prediction, allocated in RUB',
        limits=['exact accounting, not causal attribution','collinear features share contributions according to Ridge coefficients',
                'Bolt output contribution is visible, its internal features are not explained',
                'forecast dates are in 2025: no observed targets to verify accuracy']),indent=2))
    print('Exact forecast explanations:',len(summary),'forecasts;',len(feature),'feature contributions; error',max(errors),flush=True)


if __name__=='__main__':main()
