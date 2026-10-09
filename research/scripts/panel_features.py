"""Causal pooled panel examples shared by Ridge and the TabPFN pilot."""
import numpy as np

def features(residual, s, h, columns):
    return np.concatenate([residual[:,s,columns],residual[:,s-1,columns],
        residual[:,s+h-12,columns],residual[:,s,columns]-residual[:,s-1,columns]],axis=1)

def dataset(values,residual,origin,h,target,columns):
    starts=list(range(max(1,12-h),origin-h+1))
    if len(starts)<2:return None
    x=np.concatenate([features(residual,s,h,columns) for s in starts])
    y=np.concatenate([residual[:,s+h,target] for s in starts])
    scale=np.concatenate([values[:,s+h-12,target]/np.median(values[:,s+h-12,target]) for s in starts])
    return x,y,scale,features(residual,origin,h,columns),starts
