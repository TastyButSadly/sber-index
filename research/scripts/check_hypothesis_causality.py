"""Future perturbation test for the new factor methods and pooled training examples."""
import json
import numpy as np
from sklearn.linear_model import Ridge
from backtest import ROOT, load_panel
from panel_features import dataset
from forecast_common import factor_predict
from threadpoolctl import threadpool_limits

_,_,cats,values=load_panel(64);target=cats.index('Все категории');origin=17
changed=values.copy();changed[:,origin+1:]*=1000
for h in [1,2,3,6]:
    fitted=[]
    for data in [values,changed]:
        log=np.log(data);factor=np.median(log,axis=0);residual=log-factor[None,:,:]
        x,y,scale,xf,starts=dataset(data,residual,origin,h,target,np.arange(len(cats)))
        assert max(starts)+h<=origin
        with threadpool_limits(2):
            model=Ridge(alpha=30).fit(x,y,sample_weight=scale**1.5)
            local=model.predict(xf)
        fitted.append([np.exp(local+factor_predict(factor[:,target],origin,h,name))
                       for name in ['last_yoy','median3_yoy','ema_yoy','damped_yoy_trend']])
    np.testing.assert_allclose(fitted[0],fitted[1],rtol=1e-12,atol=1e-8)
print('Passed: future mutations leave all four factor forecasts and pooled Ridge predictions unchanged.')
