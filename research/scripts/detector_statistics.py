"""Paired synthetic detector tests; independent simulated sequences, not real shocks."""
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from backtest import ROOT
from experiment_statistics import holm,sign_flip


def main():
    out=ROOT/'results/statistics';out.mkdir(parents=True,exist_ok=True)
    data=pd.read_parquet(ROOT/'results/detectors_matched/synthetic_sequence_results.parquet')
    comparisons=[];rates=[]
    for scenario,d in data.groupby('scenario'):
        criteria=['any_alarm','pre_alarm']+(['hit'] if scenario!='control' else [])
        for criterion in criteria:
            for method,g in d.groupby('method'):
                success=int(g[criterion].sum());n=len(g);ci=binomtest(success,n).proportion_ci(confidence_level=.95,method='exact')
                rates.append(dict(scenario=scenario,criterion=criterion,method=method,n=n,success=success,rate=success/n,
                    ci_low=ci.low,ci_high=ci.high,p_vs_control_target_005=binomtest(success,n,.05).pvalue if scenario=='control' and criterion=='any_alarm' else np.nan,
                    scope='independent synthetic sequences only'))
            wide=d.pivot(index='sequence_id',columns='method',values=criterion).astype(int)
            for reference,candidate in itertools.combinations(wide.columns,2):
                a=wide[reference].to_numpy();b=wide[candidate].to_numpy()
                plus=int(((b==1)&(a==0)).sum());minus=int(((b==0)&(a==1)).sum())
                p=binomtest(plus,plus+minus,.5).pvalue if plus+minus else 1.
                rng=np.random.default_rng(20261009);delta=b-a;draws=[]
                for _ in range(20):
                    ix=rng.integers(0,len(delta),(500,len(delta)));draws.extend(delta[ix].mean(axis=1))
                lo,hi=np.quantile(draws,[.025,.975])
                comparisons.append(dict(scenario=scenario,criterion=criterion,reference=reference,candidate=candidate,
                    n=len(delta),delta_rate=float(delta.mean()),ci_low=lo,ci_high=hi,discordant_plus=plus,discordant_minus=minus,
                    p_exact_mcnemar=p,scope='synthetic only; calibrated thresholds fixed; paired independent sequence outcomes'))
    paired=pd.DataFrame(comparisons);paired['p_holm_synthetic_family']=holm(paired.p_exact_mcnemar)
    paired['reject_holm_005']=paired.p_holm_synthetic_family<.05
    paired.to_csv(out/'synthetic_detector_comparisons.csv',index=False)
    pd.DataFrame(rates).to_csv(out/'synthetic_detector_rates.csv',index=False)
    (out/'synthetic_protocol.json').write_text(json.dumps(dict(independent_calibration_and_evaluation_seeds=True,
        test='exact binomial test on discordant paired binary outcomes (McNemar)',
        holm_family=len(paired),bootstrap_repetitions=10000,seed=20261009,
        real_event_significance='unavailable: no labels of economic consumption breaks',
        conditional_detection_delays='no paired significance claim; sets of detected sequences differ'),indent=2))
    print('Synthetic paired comparisons:',len(paired),flush=True)
    final=ROOT/'results/final_project/detection'
    if (final/'paired_statistics.csv').exists():
        new=pd.read_csv(final/'paired_statistics.csv');panels=pd.read_csv(final/'panel_results.csv')
        for row in new.itertuples():
            d=panels[panels.scenario==row.scenario].pivot(index='panel_id',columns='method',values=row.metric)
            delta=(d[row.candidate]-d[row.reference]).to_numpy()
            assert np.isclose(delta.mean(),row.delta)
            assert np.isclose(sign_flip(delta),row.p_sign_flip)
        assert np.allclose(holm(new.p_sign_flip),new.p_holm)
        new.to_csv(out/'final_detector_panel_comparisons.csv',index=False)
        ci=[];rng=np.random.default_rng(20261009)
        for (scenario,method),g in panels.groupby(['scenario','method']):
            for metric in ['false_alarm_rate','recall','capped_delay','realized_false_discovery_fraction']:
                x=g[metric].dropna().to_numpy()
                if not len(x):continue
                ix=rng.integers(0,len(x),(10000,len(x)));lo,hi=np.quantile(x[ix].mean(axis=1),[.025,.975])
                ci.append(dict(scenario=scenario,method=method,metric=metric,panels=len(x),value=float(x.mean()),ci_low=lo,ci_high=hi,
                    scope='independent resampled panels conditional on fixed real history; not real-event accuracy'))
        pd.DataFrame(ci).to_csv(out/'final_detector_panel_rates.csv',index=False)
        (out/'final_detector_protocol.json').write_text(json.dumps(dict(status='passed',holm_family=len(new),
            independent_calibration_evaluation_seeds=True,paired_unit='resampled panel',
            threshold_target='3 notifications per 100 MO-months on no-added-change calibration',
            strict_q005='additional nominal FDR diagnostic, not a matched false-alarm-budget competitor',
            universal_FDR_guarantee=False,real_event_labels=False),indent=2))
        print('Final semi-synthetic panel comparisons:',len(new),flush=True)


if __name__=='__main__':main()
