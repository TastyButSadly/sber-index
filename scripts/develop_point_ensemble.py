import numpy as np

def factor_forecast(factor,origin,h,method):
    g=factor[12:origin+1]-factor[:origin-11]
    if not len(g):growth=0.0
    elif method=='mean1':growth=g[-1]
    elif method=='mean3':growth=np.mean(g[-3:])
    elif method=='ema':growth=np.average(g,weights=.5**np.arange(len(g)-1,-1,-1))
    else:raise ValueError(method)
    return factor[origin+h-12]+growth
