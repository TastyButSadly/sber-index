"""Paired proper interval-score comparisons; nominal levels are held fixed."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from backtest import ROOT
from experiment_statistics import KEYS,normalize,windows,sign_flip,interval,holm,save_csv_atomic


def score(g,nominal):
    a=1-nominal;y=g.actual.to_numpy();lo=g.lower.to_numpy();hi=g.upper.to_numpy()
    assert np.isfinite(lo).all() and np.isfinite(hi).all() and (lo<=hi).all()
    return hi-lo+2/a*np.maximum(lo-y,0)+2/a*np.maximum(y-hi,0)


def main():
    out=ROOT/'results/statistics';cfg=json.loads((ROOT/'configs/statistics_protocol.json').read_text());rows=[]
    for source in ['results/uncertainty/predictions.parquet','results/new_model_calibration/interval_predictions.parquet','results/final_project/interval_predictions.parquet']:
        d=pd.read_parquet(ROOT/source)
        if 'nominal' not in d:d['nominal']=.9;d['point_model']='public_ens3_mean';d['method']=d.model
        else:d['point_model']=d.model;d['method']=d.interval_method
        for (point,nominal),g in d.groupby(['point_model','nominal']):
            reference=normalize(g[g.method=='global_log']).set_index(KEYS).sort_index()
            reference['score']=score(reference,nominal)
            for method,raw in g.groupby('method'):
                if method=='global_log':continue
                candidate=normalize(raw).set_index(KEYS).sort_index()
                assert candidate.index.equals(reference.index) and np.allclose(candidate.actual,reference.actual,atol=.001)
                candidate['score']=score(candidate,nominal);candidate['delta']=candidate.score-reference.score
                candidate['score_reference']=reference.score;candidate=candidate.reset_index()
                for window,mask in windows(candidate).items():
                    if window not in ['public_origins_jun_sep','later_origins'] or not np.any(mask):continue
                    sub=candidate[mask];monthly=sub.groupby('date').delta.agg(['sum','count']).sort_index()
                    sums=monthly['sum'].to_numpy();counts=monthly['count'].to_numpy();k=len(sums);length=3
                    offsets=((monthly.index.year-monthly.index.year.min())*12+monthly.index.month-monthly.index.month[0]).to_numpy()
                    block=pd.DataFrame(dict(value=sums,n=counts,block=offsets//length)).groupby('block').sum()
                    effect=float(sub.delta.mean());lo=hi=np.nan
                    if len(block)>1:
                        se=np.sqrt(len(block)/(len(block)-1)*np.sum((block.value-effect*block.n)**2))/counts.sum()
                        margin=t.ppf(.975,len(block)-1)*se;lo=effect-margin;hi=effect+margin
                    boot_lo,boot_hi=interval(sums,counts,length,cfg['bootstrap_repetitions'],cfg['seed'])
                    rows.append(dict(source=source,point_model=point,method=method,reference='global_log',nominal=nominal,
                        window=window,n=len(sub),target_months=k,time_blocks=len(block),
                        mean_interval_score=float(sub.score.mean()),reference_interval_score=float(sub.score_reference.mean()),
                        delta_interval_score=effect,ci_time_cluster_t_low=lo,ci_time_cluster_t_high=hi,
                        descriptive_bootstrap_low=boot_lo,descriptive_bootstrap_high=boot_hi,
                        p_time_block=sign_flip(block.value.to_numpy()),p_month_sensitivity=sign_flip(sums),
                        confirmatory=False,status='exploratory; same nominal interval level; few time blocks'))
    table=pd.DataFrame(rows)
    original=pd.read_csv(ROOT/'results/new_model_calibration/interval_metrics.csv')
    checks=[]
    for row in table[table.source=='results/new_model_calibration/interval_predictions.parquet'].itertuples():
        match=original[(original.model==row.point_model)&(original.interval_method==row.method)&
                       np.isclose(original.nominal,row.nominal)&(original.window==row.window)]
        assert len(match)==1 and int(match.n.iloc[0])==row.n
        checks.append(abs(float(match.interval_score.iloc[0])-row.mean_interval_score))
    assert max(checks)<1e-6
    # One family covers observed-data point and interval-score comparisons.
    point=pd.read_csv(out/'point_comparisons.csv')
    for column in ['p_time_block','p_month_sensitivity']:
        adjusted=holm(np.r_[point[column].to_numpy(),table[column].to_numpy()])
        point[column+'_holm_point_and_interval']=adjusted[:len(point)]
        table[column+'_holm_point_and_interval']=adjusted[len(point):]
    save_csv_atomic(table,out/'interval_score_comparisons.csv')
    save_csv_atomic(point,out/'point_comparisons.csv')
    (out/'interval_protocol.json').write_text(json.dumps(dict(metric='proper interval score in RUB; lower is better',
        pairs_hold_fixed=['point model','nominal level','forecast keys'],reference='global_log',
        comparisons=len(table),combined_holm_family_size=int(point.p_time_block.notna().sum()+table.p_time_block.notna().sum()),
        coverage_tests='clustered diagnostic intervals; iid binomial p-values excluded',
        existing_interval_score_reproduction_max_rub=max(checks),
        all_exploratory=True),indent=2))
    print('Paired interval-score comparisons:',len(table),flush=True)


if __name__=='__main__':main()
