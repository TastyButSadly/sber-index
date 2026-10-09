"""Exact RUB accounting for all new fixed feature extensions."""
import json
import joblib
import numpy as np
import pandas as pd
from backtest import ROOT
from forecast_point import PointEnsemble
from develop_point_ensemble import factor_forecast


def main():
    source=ROOT/'results/combined_sources';out=source/'interpretability';out.mkdir(parents=True,exist_ok=True)
    state=joblib.load(source/'trained_extensions.joblib');point=PointEnsemble.load(ROOT/'results/point_forecast/trained_model.joblib')
    saved=pd.read_parquet(source/'future_forecasts.parquet');ids=state['ids'];summaries=[];cases=[];aggregates=[];errors=[]
    assert np.array_equal(ids,point.ids) and state['origin']==point.origin
    for variant,records in state['trained'].items():
        for h,record in records.items():
            frame=saved[(saved.model==variant+'_ema')&(saved.horizon==h)].set_index('territory_id').reindex(ids)
            fit=record['model'];xf=record['forecast_features'];reference=record['reference_features']
            baseline_local=float(fit.predict(reference[None,:])[0])
            log_contributions=(xf-reference)*fit.coef_[None,:];delta=log_contributions.sum(axis=1)
            assert np.allclose(baseline_local+delta,fit.predict(xf),atol=1e-12)
            ratio=np.ones_like(delta);np.divide(np.expm1(delta),delta,out=ratio,where=np.abs(delta)>1e-12)
            ff=factor_forecast(point.factor,point.origin_index,h,'ema')
            baseline=float(np.exp(ff+baseline_local));rub=baseline*ratio[:,None]*log_contributions/3
            bolt=(frame.bolt_prediction.to_numpy()-baseline)/3;seasonal=(frame.seasonal_prediction.to_numpy()-baseline)/3
            restored=baseline+bolt+seasonal+rub.sum(axis=1)
            error=float(np.max(np.abs(restored-frame.prediction)));errors.append(error);assert error<1e-7
            summaries.append(pd.DataFrame(dict(territory_id=ids,variant=variant,horizon=h,date=frame.date.to_numpy(),
                prediction=frame.prediction.to_numpy(),reference_prediction=baseline,bolt_contribution_rub=bolt,
                seasonal_contribution_rub=seasonal,ridge_contribution_rub=rub.sum(axis=1))))
            for j,(group,name) in enumerate(record['feature_labels']):
                aggregates.append(dict(variant=variant,horizon=h,feature_group=group,feature=name,
                    mean_abs_contribution_rub=float(np.abs(rub[:,j]).mean()),mean_contribution_rub=float(rub[:,j].mean())))
                for tid in [1333,1665,1673]:
                    i=np.flatnonzero(ids==tid)[0]
                    cases.append(dict(territory_id=tid,variant=variant,horizon=h,feature_group=group,feature=name,
                        value=float(xf[i,j]),reference_value=float(reference[j]),coefficient=float(fit.coef_[j]),
                        ensemble_contribution_rub=float(rub[i,j])))
    pd.concat(summaries,ignore_index=True).to_csv(out/'forecast_decomposition.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cases).to_csv(out/'case_feature_contributions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(aggregates).to_csv(out/'feature_summary.csv',index=False,encoding='utf-8-sig')
    (out/'checks.json').write_text(json.dumps(dict(status='passed',origin=str(point.origin.date()),method='EMA',
        forecasts=sum(len(g) for g in summaries),max_reconstruction_error_rub=max(errors),
        explanation='weighted training reference; straight path in log-linear prediction; RUB contributions divided by ensemble weight denominator 3',
        causal_attribution=False,Bolt_internal_features_explained=False,future_accuracy='unknown; no 2025 labels'),indent=2))
    print('Explained',sum(len(g) for g in summaries),'combined forecasts;',max(errors),'RUB max reconstruction error',flush=True)


if __name__=='__main__':main()
