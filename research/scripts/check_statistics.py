"""Verify custom exact tests/corrections against independent library routines."""
import json
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests
from backtest import ROOT
from experiment_statistics import sign_flip,holm


def main():
    samples=[np.array([-15.,-10.,-20.,14.,-23.,-35.]),np.array([0.,0.,0.]),np.array([-1.,2.,3.,4.])]
    for sample in samples:
        expected=permutation_test((sample,),lambda z:np.sum(z),permutation_type='samples',n_resamples=np.inf,alternative='two-sided').pvalue
        assert np.isclose(sign_flip(sample),expected)
    p=np.array([.003,.1,.02,.7,.025,1.]);expected=multipletests(p,method='holm')[1]
    assert np.allclose(holm(p),expected)
    with_nan=np.r_[p,np.nan];assert np.allclose(holm(with_nan)[:-1],expected) and np.isnan(holm(with_nan)[-1])
    d=pd.read_csv(ROOT/'results/statistics/point_comparisons.csv')
    assert np.allclose(d.delta_mae_rub,d.mae_candidate_rub-d.mae_reference_rub,atol=1e-10)
    assert ((d.p_time_block>=d.smallest_two_sided_p-1e-12)|d.p_time_block.isna()).all()
    assert not d.confirmatory.any()
    for source in ['results/combined_sources/predictions.parquet','results/architecture_extensions/predictions.parquet']:
        allp=pd.read_parquet(ROOT/source)
        for row in d[(d.source==source)&(d.window=='public_origins_jun_sep')].itertuples():
            g=allp[(allp.model==row.experiment)&allp.origin.between('2024-06-01','2024-09-01')&allp.horizon.isin([1,2,3])]
            assert len(g)==row.n
            assert np.isclose(np.abs(g.actual-g.prediction).mean(),row.mae_candidate_rub,atol=1e-8)
    (ROOT/'results/statistics/checks.json').write_text(json.dumps(dict(status='passed',
        sign_flip_matches_scipy_exact=True,holm_matches_statsmodels=True,
        reported_effects_match_paired_mae=True,point_windows_reproduced=True,
        minimum_p_resolution_respected=True,confirmatory_claims=False),indent=2))
    print('Statistical checks passed',flush=True)


if __name__=='__main__':main()
