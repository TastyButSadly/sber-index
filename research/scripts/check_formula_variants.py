"""Check the note's literal no-intercept equation and fixed 0.7/0.3 blend."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from panel_features import dataset
from forecast_common import factor_predict

ids,dates,cats,values=load_panel();target=cats.index('Все категории')
factor=np.median(np.log(values),axis=0);residual=np.log(values)-factor[None,:,:]
chosen=json.loads((ROOT/'results/hypotheses/selection.json').read_text())
rows=[]
seasonal=pd.read_parquet(ROOT/'results/hypotheses/predictions.parquet',filters=[('model','==','seasonal_growth_ema')]).set_index(['origin','horizon','territory_id'])
for o in range(11,len(dates)-1):
    for h in [1,2,3,6,12]:
        if o+h>=len(dates):continue
        for name,columns in [('uni',np.array([target])),('multi',np.arange(len(cats)))]:
            ds=dataset(values,residual,o,h,target,columns)
            if ds is None:continue
            x,y,scale,xf,_=ds;weights=scale**1.5;weights/=weights.mean()
            with threadpool_limits(2):
                local=Ridge(alpha=30,fit_intercept=False).fit(x,y,sample_weight=weights).predict(xf)
            sn=seasonal.loc[(dates[o],h)].reindex(ids).prediction.to_numpy()
            for method in [chosen['factor'],'ema_yoy']:
                forecast=np.exp(local+factor_predict(factor[:,target],o,h,method))
                for suffix,pred in [('',forecast),('_blend07',.7*forecast+.3*sn)]:
                    rows.append(pd.DataFrame({'territory_id':ids,'origin':dates[o],'date':dates[o+h],'horizon':h,
                        'model':f'literal_no_intercept_{name}_{method}{suffix}','prediction':pred,
                        'actual':values[:,o+h,target],'seasonal_value':values[:,o+h-12,target]}))
out=ROOT/'results/formula_variants';out.mkdir(parents=True,exist_ok=True)
pd.concat(rows,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
(out/'protocol.json').write_text(json.dumps({'alpha':30,'q':1.5,'fit_intercept':False,'blend_panel_weight':.7,
    'warning':'Reconstruction diagnostic added after the first evaluation; not an untouched holdout.'},indent=2),encoding='utf8')
print('Saved literal-formula comparisons; diagnostic only.')
