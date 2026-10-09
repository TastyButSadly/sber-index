"""Recover individual predictions behind previously saved tuning metrics."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from panel_features import dataset
from forecast_common import factor_predict


def main():
    out=ROOT/'results/ridge_grid';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'results/hypotheses/config.json').read_text())
    choice=json.loads((ROOT/'results/hypotheses/selection.json').read_text())['factor']
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    logs=np.log(values);factor=np.median(logs,axis=0);residual=logs-factor[None,:,:]
    frames=[]
    with threadpool_limits(2):
        for o in range(11,len(dates)-1):
            for h in cfg['horizons']:
                if o+h>=len(dates):continue
                ff=factor_predict(factor[:,target],o,h,choice)
                for mode,columns in [('uni',np.array([target])),('multi',np.arange(len(cats)))]:
                    ds=dataset(values[:,:o+1],residual[:,:o+1],o,h,target,columns)
                    if ds is None:continue
                    x,y,scale,xf,_=ds
                    for q in cfg['q_grid']:
                        w=scale**q;w/=w.mean()
                        for alpha in cfg['alpha_grid']:
                            fit=Ridge(alpha=alpha).fit(x,y,sample_weight=w)
                            frames.append(pd.DataFrame(dict(territory_id=ids,origin=dates[o],date=dates[o+h],horizon=h,
                                actual=values[:,o+h,target],seasonal_value=values[:,o+h-12,target],
                                prediction=np.exp(fit.predict(xf)+ff),model=f'{mode}_q{q:g}_a{alpha:g}')))
    p=pd.concat(frames,ignore_index=True)
    old=pd.read_csv(ROOT/'results/hypotheses/ridge_grid_metrics.csv');errors=[]
    for row in old.itertuples():
        hh=[1,2,3] if str(row.horizon)=='1_2_3' else [int(row.horizon)]
        g=p[(p.model==row.model)&p.horizon.isin(hh)]
        masks={'calibration_jan_may':g.date<=pd.Timestamp(cfg['calibration_end']),
               'replication_jun_sep':g.date.between('2024-06-01',cfg['reproduction_end']),
               'stress_oct_nov':g.date.between('2024-10-01',cfg['stress_end']),
               'december_audit':g.date>pd.Timestamp(cfg['stress_end'])}
        g=g[masks[row.window]];assert len(g)==row.n
        error=abs(float(np.abs(g.actual-g.prediction).mean())-row.mae_rub);errors.append(error)
    assert max(errors)<1e-6
    p.to_parquet(out/'predictions.parquet',index=False)
    (out/'checks.json').write_text(json.dumps(dict(status='passed',recovered_configurations=p.model.nunique(),
        original_metrics_reproduction_max_rub=max(errors),original_factor_choice=choice,
        observed_prefix_only=True,selection_status='same historical tuning, not a new independent experiment'),indent=2))
    print('Recovered',p.model.nunique(),'Ridge configurations;',len(p),'paired predictions; max metric error',max(errors),flush=True)


if __name__=='__main__':main()
