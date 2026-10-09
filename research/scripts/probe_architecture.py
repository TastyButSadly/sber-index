"""Post-audit diagnostics; these are exploratory, not new blind validation."""
import json
import numpy as np
import pandas as pd
from backtest import ROOT,load_panel
from forecast_common import factor_predict

OUT=ROOT/'results/architecture_probe'
KEYS=['territory_id','origin','date','horizon']

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    factor=np.median(np.log(values[:,:,target]),axis=0)
    source=pd.read_parquet(ROOT/'results/reference_audit/predictions.parquet')
    sources={'last':'public_factor_only','ets':'public_ets_relative','bolt':'public_bolt_base',
             'ridge':'weighted_panel_ridge','seasonal_growth':'public_snaive_growth'}
    chosen=json.loads((ROOT/'results/hypotheses/selection.json').read_text())['factor']
    def forecast(o,h,method):
        i=dates.get_loc(o);g=factor[12:i+1]-factor[:i-11]
        if method=='mean1':growth=g[-1]
        elif method=='mean3':growth=g[-3:].mean()
        elif method=='median_combo':return np.median([forecast(o,h,m) for m in ['mean1','mean3','ema_yoy']])
        else:return factor_predict(factor,i,h,method)
        return factor[i+h-12]+growth
    originals={'last':'mean3','ets':'mean1','bolt':'mean1','ridge':chosen,'seasonal_growth':'mean3'}
    cubes={};grid=[]
    for local,model in sources.items():
        d=source[(source.model==model)&(source.origin>=dates[12])].copy()
        residual=np.log(d.prediction)-np.array([forecast(o,h,originals[local]) for o,h in zip(d.origin,d.horizon)])
        for method in ['mean1','mean3','ema_yoy','damped_yoy_trend','median_combo','ORACLE_DIAGNOSTIC']:
            ff=np.array([factor[dates.get_loc(t)] if method=='ORACLE_DIAGNOSTIC' else forecast(o,h,method) for o,h,t in zip(d.origin,d.horizon,d.date)])
            out=d.copy();out.prediction=np.exp(residual+ff);out.model=local+'__'+method
            cubes[out.model.iloc[0]]=out
            for label,mask in [('public_origins_jun_sep',out.origin.between('2024-06-01','2024-09-01')),
                               ('targets_oct_nov',out.date.between('2024-10-01','2024-11-01'))]:
                g=out[mask]
                grid.append(dict(local=local,factor=method,window=label,n=len(g),mae_rub=np.mean(np.abs(g.actual-g.prediction))))
    pd.DataFrame(grid).to_csv(OUT/'factor_local_matrix.csv',index=False)
    base=source[source.model=='public_ens3_mean'].set_index(KEYS)
    candidates={
        'ridge_original':source[source.model=='weighted_panel_ridge'],
        'ridge_factor_mean1':cubes['ridge__mean1'],
        'ridge_factor_ema':cubes['ridge__ema_yoy'],
        'our_existing_ensemble':source[source.model=='ridge_seasonal_ensemble'],
    }
    correlations=[];combination=[]
    for name,frame in candidates.items():
        x=base.join(frame.set_index(KEYS)[['prediction']].rename(columns={'prediction':'candidate'}),how='inner')
        # All choices below are fixed before this probe, but after previous outcomes were seen.
        cal=x[x.index.get_level_values('date')<=pd.Timestamp('2024-05-01')]
        loss={w:np.mean(np.abs(cal.actual-((1-w)*cal.prediction+w*cal.candidate))) for w in [0,.1,.2,.3]}
        selected=min(loss,key=loss.get)
        for w in [0,.1,.2,.25,.3]:
            for label,mask in [('public_origins_jun_sep',x.index.get_level_values('origin').to_series(index=x.index).between('2024-06-01','2024-09-01')),
                               ('targets_oct_nov',x.index.get_level_values('date').to_series(index=x.index).between('2024-10-01','2024-11-01'))]:
                g=x[mask.to_numpy()]
                pred=(1-w)*g.prediction+w*g.candidate
                combination.append(dict(candidate=name,weight=w,window=label,n=len(g),mae_rub=np.mean(np.abs(g.actual-pred)),
                    calibration_selected_weight=selected,calibration_mae=loss.get(w,np.nan)))
        g=x[x.index.get_level_values('origin').to_series(index=x.index).between('2024-06-01','2024-09-01').to_numpy()]
        e=g.prediction-g.actual;ec=g.candidate-g.actual;tail=np.abs(e)>=np.quantile(np.abs(e),.9)
        correlations.append(dict(candidate=name,error_corr=e.corr(ec),absolute_error_corr=np.abs(e).corr(np.abs(ec)),
            share_candidate_better=np.mean(np.abs(ec)<np.abs(e)),tail_base_mae=np.abs(e[tail]).mean(),tail_candidate_mae=np.abs(ec[tail]).mean()))
    pd.DataFrame(combination).to_csv(OUT/'fixed_combination.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUT/'complementarity.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps({'status':'exploratory after observing history',
        'no_new_training':True,'weight_calibration':'target dates through May 2024',
        'oracle':'future factor only for diagnostics; never forecast or eligible ensemble'},indent=2))
    print(pd.DataFrame(grid).query('window == "public_origins_jun_sep"').pivot(index='local',columns='factor',values='mae_rub').round(2).to_string())
    print(pd.DataFrame(combination).query('window == "public_origins_jun_sep"').to_string(index=False))
    print(pd.DataFrame(correlations).round(3).to_string(index=False))

if __name__=='__main__':main()
