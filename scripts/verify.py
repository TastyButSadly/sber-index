"""Verify reproduced report metrics against the immutable evidence snapshot."""
import hashlib
import json
import numpy as np
import pandas as pd
from backtest import ROOT


def main():
    out = ROOT/'results/final_project'
    checks = {}
    for name, keys in [('window_metrics.csv',['model','window']), ('interval_window_metrics.csv',['interval_method','nominal','window'])]:
        actual = pd.read_csv(out/name).sort_values(keys).reset_index(drop=True)
        expected = pd.read_csv(ROOT/'evidence/final_project'/name).sort_values(keys).reset_index(drop=True)
        assert len(actual) == len(expected)
        assert actual[keys].equals(expected[keys])
        for column in actual.select_dtypes(include='number'):
            np.testing.assert_allclose(actual[column], expected[column], rtol=1e-5, atol=.1 if 'rub' in column else 1e-5)
        checks[name] = 'passed'
    for name in ['predictions.parquet','interval_predictions.parquet']:
        d = pd.read_parquet(out/name)
        key = ['territory_id','origin','date','horizon','model']
        if 'nominal' in d: key += ['nominal','interval_method']
        assert not d.duplicated(key).any()
        assert np.isfinite(d.prediction).all() and (d.prediction > 0).all()
    checks['unique_forecast_keys'] = 'passed'
    detector = out/'detection/metrics.csv'
    if detector.exists():
        actual = pd.read_csv(detector).sort_values(['scenario','method']).reset_index(drop=True)
        expected = pd.read_csv(ROOT/'evidence/final_project/detection/metrics.csv').sort_values(['scenario','method']).reset_index(drop=True)
        assert actual[['scenario','method']].equals(expected[['scenario','method']])
        for column in actual.select_dtypes(include='number'):
            np.testing.assert_allclose(actual[column],expected[column],rtol=1e-8,atol=1e-8,equal_nan=True)
        checks['detector_summary'] = 'passed'
    checks['report_pdf_sha256'] = hashlib.sha256((ROOT/'deliverables/hackathon_report_latex/main.pdf').read_bytes()).hexdigest()
    (ROOT/'results/reproduction_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)


if __name__ == '__main__':
    main()
