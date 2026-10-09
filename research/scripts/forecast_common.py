"""Causal forecast of the cross-sectional log-consumption factor."""
import numpy as np

def factor_predict(factor, origin, h, method):
    if origin < 12:
        slope = np.polyfit(np.arange(origin+1), factor[:origin+1], 1)[0]
        return factor[origin+h-12] + 12*slope
    yoy = factor[12:origin+1] - factor[:origin-11]
    if method == 'last_yoy': growth = yoy[-1]
    elif method == 'median3_yoy': growth = np.median(yoy[-3:])
    elif method == 'ema_yoy':
        weights = 0.5 ** np.arange(len(yoy)-1, -1, -1)
        growth = np.average(yoy, weights=weights)
    elif method == 'damped_yoy_trend':
        growth = yoy[-1]
        if len(yoy) >= 3:
            growth += 0.5*min(h,3)*np.median(np.diff(yoy[-4:]))
    else: raise ValueError(method)
    return factor[origin+h-12] + growth
