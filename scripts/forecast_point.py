"""Fit and run Bolt + pooled Ridge + seasonal local forecasts, without future labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from backtest import ROOT,load_panel
from panel_features import dataset
from develop_point_ensemble import factor_forecast


class PointEnsemble:
    def __init__(self, horizons=(1,2,3), alpha=3.0, weight_power=1.5):
        self.horizons=tuple(horizons)
        self.alpha=alpha
        self.weight_power=weight_power

    @classmethod
    def load(cls,path):
        state=joblib.load(path)
        model=cls(state['horizons'],state['alpha'],state['weight_power'])
        model.__dict__.update(state)
        return model

    def fit(self, observed, ids, dates, categories):
        """observed contains only complete positive monthly observations through origin."""
        values=np.asarray(observed,dtype=float)
        dates=pd.DatetimeIndex(dates)
        if values.shape!=(len(ids),len(dates),len(categories)):
            raise ValueError('Input dimensions do not match IDs, dates and categories')
        if not dates.equals(pd.date_range(dates[0],periods=len(dates),freq='MS')):
            raise ValueError('Need consecutive monthly observations')
        if len(dates)<14 or not np.isfinite(values).all() or (values<=0).any():
            raise ValueError('Need at least 14 complete positive monthly observations')
        if not self.horizons or any(h<1 or h>12 for h in self.horizons):
            raise ValueError('Supported horizons are 1 through 12 months')
        self.ids=np.asarray(ids)
        self.categories=list(categories)
        self.origin=dates[-1]
        self.origin_index=len(dates)-1
        target=self.categories.index('Все категории')
        logs=np.log(values)
        factor=np.median(logs,axis=0)
        residual=logs-factor[None,:,:]
        self.factor=factor[:,target]
        self.bolt_context=residual[:,:,target].copy()
        changes=logs[:,12:,target]-logs[:,:-12,target]
        # Individual YoY mean3, represented in the same local/factor space as audit.
        growth=np.mean(changes[:,-3:],axis=1)
        self.ridge={};self.seasonal_local={}
        with threadpool_limits(2):
            for h in self.horizons:
                ds=dataset(values,residual,self.origin_index,h,target,np.arange(len(categories)))
                if ds is None:raise ValueError(f'Not enough matured training examples for horizon {h}')
                x,y,scale,xf,_=ds
                w=scale**self.weight_power;w/=w.mean()
                model=Ridge(alpha=self.alpha).fit(x,y,sample_weight=w)
                self.ridge[h]=(model,xf)
                self.seasonal_local[h]=logs[:,self.origin_index+h-12,target]+growth-factor_forecast(self.factor,self.origin_index,h,'mean3')
        return self

    def predict(self, checkpoint=None):
        import torch
        from chronos import BaseChronosPipeline
        torch.set_num_threads(2)
        checkpoint=checkpoint or ROOT/'models/chronos-bolt-base'
        pipe=BaseChronosPipeline.from_pretrained(str(checkpoint),device_map='cpu',torch_dtype=torch.float32)
        local=[]
        for start in range(0,len(self.ids),64):
            context=[torch.tensor(r,dtype=torch.float32) for r in self.bolt_context[start:start+64]]
            q,_=pipe.predict_quantiles(context,prediction_length=max(self.horizons),quantile_levels=[.5])
            local.append(q[...,0].numpy())
        bolt=np.vstack(local)
        frames=[]
        for method in ('mean1','ema'):
            for h in self.horizons:
                f=factor_forecast(self.factor,self.origin_index,h,method)
                ridge,xf=self.ridge[h]
                components={'bolt':np.exp(bolt[:,h-1]+f),
                            'ridge':np.exp(ridge.predict(xf)+f),
                            'seasonal':np.exp(self.seasonal_local[h]+f)}
                p=np.mean(list(components.values()),axis=0)
                frames.append(pd.DataFrame(dict(territory_id=self.ids,origin=self.origin,
                    date=self.origin+pd.DateOffset(months=h),horizon=h,
                    model='bolt_ridge_seasonal_'+method,prediction=p,
                    **{name+'_prediction':value for name,value in components.items()})))
        result=pd.concat(frames,ignore_index=True)
        if not np.isfinite(result.prediction).all() or (result.prediction<=0).any():
            raise ValueError('Invalid ensemble prediction')
        return result


def fit_at(values,ids,dates,categories,origin_index,horizons):
    return PointEnsemble(horizons).fit(values[:,:origin_index+1],ids,dates[:origin_index+1],categories)

