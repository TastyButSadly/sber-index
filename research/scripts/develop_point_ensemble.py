"""Build deployable factor/local ensembles and compare matured-label selectors."""
import json
import numpy as np
import pandas as pd
from backtest import ROOT,load_panel

KEYS=['territory_id','origin','date','horizon']
OUT=ROOT/'results/point_ensemble'

def factor_forecast(factor,origin,h,method):
    g=factor[12:origin+1]-factor[:origin-11]
    if not len(g):growth=0.0
    elif method=='mean1':growth=g[-1]
    elif method=='mean3':growth=np.mean(g[-3:])
    elif method=='ema':growth=np.average(g,weights=.5**np.arange(len(g)-1,-1,-1))
    else:raise ValueError(method)
    return factor[origin+h-12]+growth

def select_past(d,o,cfg):
    past=d[(d.date<=o)&(d.origin<o)]
    months=sorted(past.date.unique())
    if len(months)<cfg['minimum_selection_target_months']:return None
    return past[past.date.isin(months[-cfg['selection_lookback_target_months']:])]

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    cfg=json.loads((ROOT/'configs/point_ensemble.json').read_text())
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    factor=np.median(np.log(values[:,:,target]),axis=0)
    source=pd.read_parquet(ROOT/'results/reference_audit/predictions.parquet')
    def get(name):return source[source.model==name].set_index(KEYS).sort_index()
    reference=get('public_ens3_mean')
    # Keep only complete shared keys, including the fixed pooled Ridge candidate.
    ridge=get('weighted_panel_ridge_public_factor_mean1')
    idx=reference.index.intersection(ridge.index)
    d=reference.loc[idx,['actual','seasonal_value']].reset_index()
    forecasts={};methods=cfg['factor_methods']
    for local,name,old_method in [('ets','public_ets_relative','mean1'),('bolt','public_bolt_base','mean1'),
                                  ('ridge','weighted_panel_ridge_public_factor_mean1','mean1'),
                                  ('seasonal','public_snaive_growth','mean3')]:
        pred=get(name).prediction.reindex(idx).to_numpy()
        residual=pred.copy()
        for (o,h),g in d.groupby(['origin','horizon']):
            pos=g.index.to_numpy();i=dates.get_loc(o)
            residual[pos]=np.log(pred[pos])-factor_forecast(factor,i,h,old_method)
        for method in methods:
            p=np.empty(len(d))
            for (o,h),g in d.groupby(['origin','horizon']):
                pos=g.index.to_numpy();p[pos]=np.exp(residual[pos]+factor_forecast(factor,dates.get_loc(o),h,method))
            forecasts[local+'_'+method]=p
    forecasts['seasonal_local_ema']=get('seasonal_growth_ema').prediction.reindex(idx).to_numpy()
    forecasts['previous_ensemble']=get('ridge_seasonal_ensemble').prediction.reindex(idx).to_numpy()
    library={'reference_ens3':reference.prediction.reindex(idx).to_numpy()}
    for method in methods:
        library['ens3_factor_'+method]=np.mean([forecasts['ets_'+method],forecasts['bolt_'+method],forecasts['seasonal_'+method]],axis=0)
        library['ens4_factor_'+method]=np.mean([forecasts['ets_'+method],forecasts['bolt_'+method],forecasts['seasonal_'+method],forecasts['ridge_'+method]],axis=0)
        library['bolt_ridge_seasonal_'+method]=np.mean([forecasts['bolt_'+method],forecasts['ridge_'+method],forecasts['seasonal_'+method]],axis=0)
    library['mixed_ets_bolt_ema']=np.mean([forecasts['ets_mean1'],forecasts['bolt_ema'],forecasts['seasonal_mean3']],axis=0)
    library['mixed_ridge_ema']=np.mean([forecasts['ets_mean1'],forecasts['bolt_ema'],forecasts['seasonal_mean3'],forecasts['ridge_ema']],axis=0)
    library['local_ema_ens4']=np.mean([forecasts['ets_ema'],forecasts['bolt_ema'],forecasts['seasonal_local_ema'],forecasts['ridge_ema']],axis=0)
    baseline=library['reference_ens3']
    for w in cfg['blend_weights'][1:]:library[f'reference_plus_previous_{w:g}']=(1-w)*baseline+w*forecasts['previous_ensemble']
    # Same candidate family at every origin; selections use only already observed targets.
    candidate_names=list(library)
    all_fixed=[]
    for name,p in library.items():
        full=d.copy();full['model']=name;full['prediction']=p;all_fixed.append(full)
    pd.concat(all_fixed,ignore_index=True).to_parquet(OUT/'all_candidate_predictions.parquet',index=False)
    eval_origins=sorted(d.loc[d.origin>=pd.Timestamp(cfg['first_evaluation_origin']),'origin'].unique())
    protocol=[];dynamic={name:baseline.copy() for name in ['past_select_shrunk','past_top3_mean','past_horizon_select_shrunk']}
    for raw_o in eval_origins:
        o=pd.Timestamp(raw_o);past=select_past(d,o,cfg);test=d.origin==o
        if past is None:continue
        losses={name:float(np.mean(np.abs(d.loc[past.index,'actual']-library[name][past.index]))) for name in candidate_names}
        chosen=min(losses,key=losses.get);top=sorted(losses,key=losses.get)[:3]
        s=cfg['selection_shrinkage']
        dynamic['past_select_shrunk'][test]=(1-s)*baseline[test]+s*library[chosen][test]
        dynamic['past_top3_mean'][test]=np.mean([library[n][test] for n in top],axis=0)
        protocol.append(dict(origin=str(o.date()),model='past_select_shrunk',selected=chosen,
            last_label=str(past.date.max().date()),past_target_months=past.date.nunique(),losses=losses))
        for h in cfg['horizons']:
            ph=past[past.horizon==h];th=test&(d.horizon==h)
            hl={name:float(np.mean(np.abs(ph.actual-library[name][ph.index]))) for name in candidate_names}
            best=min(hl,key=hl.get)
            dynamic['past_horizon_select_shrunk'][th]=(1-s)*baseline[th]+s*library[best][th]
            protocol.append(dict(origin=str(o.date()),horizon=h,model='past_horizon_select_shrunk',selected=best,
                last_label=str(ph.date.max().date()),past_target_months=ph.date.nunique(),losses=hl))
    library.update(dynamic)
    outputs=[];metrics=[]
    for name,p in library.items():
        frame=d.copy();frame['model']=name;frame['prediction']=p
        frame=frame[frame.origin>=pd.Timestamp(cfg['first_evaluation_origin'])]
        outputs.append(frame)
        for window,mask in [('public_origins_jun_sep',frame.origin.between('2024-06-01','2024-09-01')),
                            ('later_origins',frame.origin>pd.Timestamp('2024-09-01')),
                            ('targets_oct_nov',frame.date.between('2024-10-01','2024-11-01'))]:
            g=frame[mask]
            metrics.append(dict(model=name,window=window,n=len(g),mae_rub=float(np.mean(np.abs(g.actual-g.prediction)))))
    pd.concat(outputs,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'selection_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    # Fixed early selection is a separate reproducible candidate; no public score used for it.
    calibration=d[d.date<=pd.Timestamp('2024-05-01')]
    scores={n:float(np.mean(np.abs(calibration.actual-library[n][calibration.index]))) for n in candidate_names}
    early=min(scores,key=scores.get)
    (OUT/'early_selection.json').write_text(json.dumps({'chosen':early,'target_cutoff':'2024-05-01','losses':scores},indent=2))
    print(pd.DataFrame(metrics).pivot(index='model',columns='window',values='mae_rub').sort_values('public_origins_jun_sep').round(2).to_string())
    print('Early choice:',early,flush=True)

if __name__=='__main__':main()
