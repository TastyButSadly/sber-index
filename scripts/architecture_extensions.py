import numpy as np
KEYS=["territory_id","origin","date","horizon"]

def structural_features(values,residual,s,h,target):
    total=values[:,:,target]
    columns=[i for i in range(values.shape[2]) if i!=target]
    rest=total-values[:,:,columns].sum(axis=2)
    if (rest<=0).any():raise ValueError('Nonpositive remainder: cannot form log-ratios')
    parts=np.concatenate([values[:,:,columns],rest[:,:,None]],axis=2)
    ratio=np.log(parts/total[:,:,None])
    return np.c_[ratio[:,s],ratio[:,s]-ratio[:,s-3],ratio[:,s+h-12]]
