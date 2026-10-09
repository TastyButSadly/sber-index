"""Check future-input invariance, matured labels and external publication guard."""
import numpy as np
import pandas as pd
from backtest import ROOT,load_panel
from run_uncertainty import state_features
from build_forecast_manifest import available_external

ids,dates,cats,values=load_panel(64);target=cats.index('Все категории');origin=17
changed=values.copy();changed[:,origin+1:]*=1000
for h in [1,2,3]:
    a=state_features(values,origin,target,h,np.zeros(len(ids)),dates)
    b=state_features(changed,origin,target,h,np.zeros(len(ids)),dates)
    np.testing.assert_array_equal(a,b)
fold=pd.read_csv(ROOT/'results/uncertainty/fold_protocol.csv')
for c in ['origin','train_target_end','calibration_start','calibration_target_end']:fold[c]=pd.to_datetime(fold[c])
assert (fold.train_target_end<fold.calibration_start).all()
assert (fold.calibration_target_end<=fold.origin).all()
external=pd.DataFrame({'available_at':['2024-06-01','2024-08-01'],'value':[1,2]})
assert available_external(external,'2024-07-01').value.tolist()==[1]
try:available_external(pd.DataFrame({'value':[1]}),'2024-07-01')
except ValueError:pass
else:raise AssertionError('Unknown publication timestamp was accepted')
p=pd.read_parquet(ROOT/'results/uncertainty/predictions.parquet')
assert not p.duplicated(['model','territory_id','origin','date','horizon']).any()
assert np.isfinite(p[['lower','upper']].to_numpy()).all()
assert ((p.lower>0)&(p.lower<=p.prediction)&(p.prediction<=p.upper)).all()
base=pd.read_parquet(ROOT/'results/reference_audit/predictions.parquet',filters=[('model','==','public_ens3_mean')])
joined=p.merge(base[['territory_id','origin','date','horizon','prediction']],on=['territory_id','origin','date','horizon'],suffixes=('','_ref'),validate='many_to_one')
np.testing.assert_array_equal(joined.prediction,joined.prediction_ref)
print('Passed: future-invariant features, matured/disjoint labels, publication guard, valid intervals and unchanged point forecasts.')
