"""Check deployment reproduction and exclusion of future values and selection labels."""
import json
import joblib
import numpy as np
import pandas as pd
from backtest import ROOT,load_panel
from forecast_point import PointEnsemble,fit_at
from develop_point_ensemble import select_past


def main():
    out=ROOT/'results/point_ensemble'
    keys=['territory_id','origin','date','horizon','model']
    cached=pd.read_parquet(out/'predictions.parquet').set_index(keys)
    deployed=pd.read_parquet(ROOT/'results/point_forecast_verification/predictions.parquet').set_index(keys)
    expected=cached.prediction.reindex(deployed.index)
    assert expected.notna().all() and deployed.index.is_unique
    delta=np.abs(expected-deployed.prediction)
    assert np.allclose(expected,deployed.prediction,rtol=1e-6,atol=.01),float(delta.max())
    ids,dates,categories,values=load_panel()
    o=dates.get_loc('2024-09-01')
    clean=fit_at(values,ids,dates,categories,o,[1,2,3])
    perturbed=values.copy();perturbed[:,o+1:]*=1000
    altered=fit_at(perturbed,ids,dates,categories,o,[1,2,3])
    assert np.array_equal(clean.factor,altered.factor)
    assert np.array_equal(clean.bolt_context,altered.bolt_context)
    for h in clean.horizons:
        assert np.array_equal(clean.seasonal_local[h],altered.seasonal_local[h])
        left,x=clean.ridge[h];right,y=altered.ridge[h]
        assert np.array_equal(left.predict(x),right.predict(y))
    cfg=json.loads((ROOT/'configs/point_ensemble.json').read_text())
    d=cached.reset_index();d=d[d.model=='reference_ens3'].reset_index(drop=True)
    origin=pd.Timestamp('2024-09-01')
    before=select_past(d,origin,cfg)
    assert before is not None
    corrupt=d.copy();corrupt.loc[corrupt.date>origin,'actual']*=1000
    after=select_past(corrupt,origin,cfg)
    pd.testing.assert_frame_equal(before,after)
    # Save a portable state dictionary, also replacing the first CLI run's old class pickle.
    path=ROOT/'results/point_forecast_verification/trained_model.joblib'
    joblib.dump(clean.__dict__,path)
    restored=PointEnsemble.load(path)
    assert np.array_equal(restored.bolt_context,clean.bolt_context)
    for h in clean.horizons:
        a,x=clean.ridge[h];b,y=restored.ridge[h]
        assert np.array_equal(a.predict(x),b.predict(y))
    result=dict(status='passed',deployment_max_absolute_difference_rub=float(delta.max()),
        checked_forecasts=len(deployed),future_values_multiplier=1000,
        future_values_unchanged_predictions=True,future_labels_excluded_from_selection=True,
        saved_model_reload=True)
    (out/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
