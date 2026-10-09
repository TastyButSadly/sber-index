"""Audit saved experiments with paired, dependence-aware exploratory statistics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import t
from backtest import ROOT

KEYS=['territory_id','origin','date','horizon']
OUT=ROOT/'results/statistics'


def save_csv_atomic(frame,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temporary,index=False,encoding='utf-8-sig',chunksize=250)
    temporary.replace(path)


def holm(values):
    p=np.asarray(values,dtype=float);result=np.full(len(p),np.nan);valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid])]
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(order)-np.arange(len(order)))))
    return result


def sign_flip(sums):
    sums=np.asarray(sums,dtype=float);k=len(sums)
    if k<2:return np.nan
    observed=abs(sums.sum());tol=max(1e-10,observed*1e-12)
    if k<=18:
        signs=np.array(list(itertools.product([-1,1],repeat=k)),dtype=float)
        return float(np.mean(np.abs(signs@sums)>=observed-tol))
    rng=np.random.default_rng(20261009);signs=rng.choice([-1.,1.],(9999,k))
    return float((1+np.sum(np.abs(signs@sums)>=observed-tol))/10000)


def interval(sums,counts,length,repetitions,seed):
    k=len(sums)
    if k<max(3,2*length):return np.nan,np.nan
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,k-length+1,(repetitions,math.ceil(k/length)))
    indices=(starts[:,:,None]+np.arange(length)[None,None,:]).reshape(repetitions,-1)[:,:k]
    draws=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    return tuple(np.quantile(draws,[.025,.975]))


def normalize(frame):
    frame=frame.copy()
    for c in ['origin','date']:frame[c]=pd.to_datetime(frame[c])
    if frame.duplicated(KEYS).any():
        duplicates=frame[frame.duplicated(KEYS,keep=False)]
        for c in ['actual','prediction']:
            if c in frame and (duplicates.groupby(KEYS)[c].nunique()>1).any():
                raise ValueError('Conflicting duplicate forecasts: '+c)
        frame=frame.drop_duplicates(KEYS)
    return frame


def windows(g):
    masks={'public_origins_jun_sep':g.origin.between('2024-06-01','2024-09-01'),
            'later_origins':g.origin>pd.Timestamp('2024-09-01'),
            'targets_jun_sep':g.date.between('2024-06-01','2024-09-01'),
            'targets_oct_nov':g.date.between('2024-10-01','2024-11-01'),
            'targets_dec':g.date==pd.Timestamp('2024-12-01'),
            'targets_jan_may':g.date.between('2024-01-01','2024-05-01'),
            'all_available':np.ones(len(g),dtype=bool)}
    output={name:mask&g.horizon.isin([1,2,3]) for name,mask in masks.items()}
    for h in sorted(set(g.horizon)-{1,2,3}):
        output.update({name+f'/h{h}':mask&(g.horizon==h) for name,mask in masks.items()})
    return output


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/statistics_protocol.json').read_text())
    (OUT/'protocol.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
    geo=pd.read_parquet(ROOT/'data/processed/municipalities_2024.parquet').drop_duplicates('territory_id').set_index('territory_id').region_code
    cache={};catalog=[];results=[];monthly=[];seen={};aliases=[]
    def read(path):
        if path not in cache:
            fields=pq.read_schema(ROOT/path).names
            columns=[c for c in KEYS+['model','actual','prediction','point_method','interval_method','nominal','lower','upper'] if c in fields]
            cache[path]=pd.read_parquet(ROOT/path,columns=columns)
        return cache[path]
    def base(path,model):return normalize(read(path).query('model==@model'))
    def compare(source,name,candidate,reference,reference_name):
        candidate=normalize(candidate);reference=normalize(reference)
        # Scoring is always restricted to exact shared forecast keys.
        g=candidate.merge(reference[KEYS+['actual','prediction']],on=KEYS,suffixes=('','_reference'),validate='one_to_one')
        if g.empty:
            catalog.append(dict(source=source,experiment=name,status='no shared evaluation keys',comparator=reference_name));return
        assert np.allclose(g.actual,g.actual_reference,rtol=0,atol=.001)
        g=g.sort_values(KEYS).reset_index(drop=True)
        loss=np.abs(g.actual-g.prediction);ref_loss=np.abs(g.actual-g.prediction_reference)
        g['delta']=loss-ref_loss;g['region_code']=g.territory_id.map(geo)
        catalog.append(dict(source=source,experiment=name,status='paired forecast statistics recorded',comparator=reference_name,
            available_predictions=len(candidate),paired_predictions=len(g)))
        for window,mask in windows(g).items():
            sub=g.loc[mask].copy()
            if sub.empty:continue
            fingerprint=hashlib.sha256(pd.util.hash_pandas_object(sub[KEYS+['actual','prediction','prediction_reference']],index=False).values.tobytes()).hexdigest()
            identity=(window,fingerprint)
            if identity in seen:
                aliases.append(dict(source=source,experiment=name,comparator=reference_name,window=window,canonical_test_id=seen[identity]));continue
            test_id=f'test_{len(results)+1:05d}';seen[identity]=test_id
            months=sub.groupby('date').delta.agg(['sum','count']).sort_index()
            sums=months['sum'].to_numpy();counts=months['count'].to_numpy();k=len(months)
            length=max(cfg['time_block_months'],int(sub.horizon.max()))
            offsets=((months.index.year-months.index.year.min())*12+months.index.month-months.index.month[0]).to_numpy()
            blocks=pd.DataFrame(dict(value=sums,block=offsets//length)).groupby('block').value.sum().to_numpy()
            block_counts=pd.DataFrame(dict(value=counts,block=offsets//length)).groupby('block').value.sum().to_numpy()
            effect=float(sums.sum()/counts.sum());cluster_lo=cluster_hi=np.nan
            if len(blocks)>1:
                scores=blocks-effect*block_counts
                se=float(np.sqrt(len(blocks)/(len(blocks)-1)*np.sum(scores**2))/counts.sum())
                margin=t.ppf(.975,df=len(blocks)-1)*se
                cluster_lo=effect-margin;cluster_hi=effect+margin
            block_lo,block_hi=interval(sums,counts,length,cfg['bootstrap_repetitions'],cfg['seed'])
            month_lo,month_hi=interval(sums,counts,1,cfg['bootstrap_repetitions'],cfg['seed'])
            regions=sub.groupby('region_code').delta.agg(['sum','count']);region_lo=region_hi=np.nan
            if len(regions)>=10 and sub.region_code.notna().all():
                rng=np.random.default_rng(cfg['seed']);ix=rng.integers(0,len(regions),(cfg['bootstrap_repetitions'],len(regions)))
                draws=regions['sum'].to_numpy()[ix].sum(axis=1)/regions['count'].to_numpy()[ix].sum(axis=1)
                region_lo,region_hi=np.quantile(draws,[.025,.975])
            c_mae=float(np.abs(sub.actual-sub.prediction).mean());r_mae=float(np.abs(sub.actual-sub.prediction_reference).mean())
            results.append(dict(test_id=test_id,source=source,experiment=name,comparator=reference_name,window=window,n=len(sub),
                municipalities=sub.territory_id.nunique(),target_months=k,time_blocks=len(blocks),block_months=length,
                horizons=','.join(map(str,sorted(sub.horizon.unique()))),
                mae_candidate_rub=c_mae,mae_reference_rub=r_mae,delta_mae_rub=c_mae-r_mae,
                relative_improvement_percent=100*(r_mae-c_mae)/r_mae,
                ci_time_block_low=block_lo,ci_time_block_high=block_hi,ci_month_low=month_lo,ci_month_high=month_hi,
                ci_time_cluster_t_low=cluster_lo,ci_time_cluster_t_high=cluster_hi,
                ci_region_conditional_low=region_lo,ci_region_conditional_high=region_hi,
                p_time_block=sign_flip(blocks),p_month_sensitivity=sign_flip(sums),
                smallest_two_sided_p=2**(1-len(blocks)) if len(blocks)>1 else np.nan,
                status='exploratory; insufficient time blocks' if len(blocks)<6 else 'exploratory; assumptions unverified',
                confirmatory=False,confidence_interval_scope='exploratory; cluster-t assumes independent approximately normal block scores; bootstrap descriptive; no simultaneous CI correction'))
            for date,row in months.iterrows():monthly.append(dict(test_id=test_id,date=str(date.date()),n=int(row['count']),delta_mae_rub=row['sum']/row['count']))
    ordinary={
        'baselines/predictions.parquet':('baselines/predictions.parquet','ridge_seasonal_ensemble'),
        'hypotheses/predictions.parquet':('hypotheses/predictions.parquet','ridge_seasonal_ensemble'),
        'ridge_grid/predictions.parquet':('hypotheses/predictions.parquet','weighted_panel_ridge'),
        'reference_audit/predictions.parquet':('reference_audit/predictions.parquet','public_ens3_mean'),
        'formula_variants/predictions.parquet':('hypotheses/predictions.parquet','ridge_seasonal_ensemble'),
        'point_ensemble/all_candidate_predictions.parquet':('point_ensemble/all_candidate_predictions.parquet','reference_ens3'),
        'point_ensemble/predictions.parquet':('point_ensemble/predictions.parquet','reference_ens3'),
        'next_forecast_checks/predictions.parquet':('next_forecast_checks/predictions.parquet','anchor_ema'),
        'architecture_extensions/predictions.parquet':('architecture_extensions/predictions.parquet','anchor'),
        'architecture_extensions/ablation_predictions.parquet':('architecture_extensions/predictions.parquet','anchor'),
        'generalization/predictions.parquet':('generalization/predictions.parquet','pooled_structure_macro'),
        'full_calendar/predictions.parquet':('full_calendar/predictions.parquet','structure_macro'),
        'residual_economy/predictions.parquet':('residual_economy/predictions.parquet','structure_macro_prices_wages'),
        'final_project/predictions.parquet':('final_project/predictions.parquet','structure_macro_prices_wages'),
        'tabpfn35_pilot/predictions.parquet':('tabpfn35_pilot/predictions.parquet','ridge_same_training_sample'),
        'tabpfn35_panel_pilot/predictions.parquet':('tabpfn35_panel_pilot/predictions.parquet','ridge_same_training_sample'),
        'prophet_smoke/predictions.parquet':('baselines/predictions.parquet','seasonal_naive'),
    }
    for folder in ['chronos_cross','chronos_multivariate','chronos_residual','chronos_univariate','chronos_smoke']:
        ordinary[f'{folder}/predictions.parquet']=('hypotheses/predictions.parquet','ridge_seasonal_ensemble')
    processed=set()
    for relative,(ref_path,ref_name) in ordinary.items():
        path='results/'+relative
        if not (ROOT/path).exists():continue
        reference=base('results/'+ref_path,ref_name);processed.add(path)
        for name,g in read(path).groupby('model'):
            if path=='results/'+ref_path and name==ref_name:continue
            compare(path,name,g,reference,ref_name)
        print('Statistics:',relative,flush=True)
        cache.clear();gc.collect()
    path='results/combined_sources/predictions.parquet'
    if (ROOT/path).exists():
        processed.add(path)
        for ref in ['anchor','composition','macro_interactions']:
            reference=base(path,ref)
            for name,g in read(path).groupby('model'):
                if name!=ref:compare(path,name,g,reference,ref)
        cache.clear();gc.collect()
    path='results/new_model_calibration/corrected_point_predictions.parquet';processed.add(path)
    for (name,method),g in read(path).groupby(['model','point_method']):
        if method=='original':continue
        reference=normalize(read(path)[(read(path).model==name)&(read(path).point_method=='original')])
        compare(path,name+'/'+method,g,reference,name+'/original')
    cache.clear();gc.collect()
    path='results/contest_horizons/predictions.parquet';processed.add(path)
    d=read(path)
    for h,hh in d.groupby('horizon'):
        for ref in ['prophet_monthly','seasonal_naive']:
            reference=normalize(hh[hh.model==ref])
            for name,g in hh.groupby('model'):
                if name.startswith('portfolio_'):compare(path,name+f'/h{h}',g,reference,ref+f'/h{h}')
    cache.clear();gc.collect()
    path='results/hypothesis_analysis/selected_ensemble_predictions.parquet';processed.add(path)
    if (ROOT/path).exists():
        for name,g in read(path).groupby('model'):compare(path,'selected_foundation_ensemble',g,base('results/hypotheses/predictions.parquet','ridge_seasonal_ensemble'),'ridge_seasonal_ensemble')
    table=pd.DataFrame(results)
    for column in ['p_time_block','p_month_sensitivity']:table[column+'_holm_all']=holm(table[column])
    table['reject_time_block_holm_005']=table.p_time_block_holm_all<cfg['alpha']
    table['significant_improvement_time_block_holm_005']=table.reject_time_block_holm_005&(table.delta_mae_rub<0)
    save_csv_atomic(table,OUT/'point_comparisons.csv')
    pd.DataFrame(monthly).to_csv(OUT/'monthly_paired_effects.csv',index=False)
    pd.DataFrame(aliases).to_csv(OUT/'comparison_aliases.csv',index=False)
    # Coverage is clustered by target month; no binomial test treating all MO as iid.
    coverage=[]
    for path in ['results/uncertainty/predictions.parquet','results/new_model_calibration/interval_predictions.parquet','results/final_project/interval_predictions.parquet']:
        if not (ROOT/path).exists():continue
        processed.add(path);d=read(path)
        groups=['model']+(['interval_method','nominal'] if 'nominal' in d else [])
        for label,g in d.groupby(groups):
            g=normalize(g);nominal=float(g.nominal.iloc[0]) if 'nominal' in g else .9
            name='/'.join(map(str,label if isinstance(label,tuple) else [label]))
            hit=((g.actual>=g.lower)&(g.actual<=g.upper)).astype(float)
            for window,mask in windows(g).items():
                if window not in ['public_origins_jun_sep','later_origins'] or not np.any(mask):continue
                sub=g[mask];months=pd.DataFrame(dict(date=sub.date,hit=hit[mask])).groupby('date').hit.agg(['sum','count'])
                lo,hi=interval(months['sum'].to_numpy(),months['count'].to_numpy(),3,cfg['bootstrap_repetitions'],cfg['seed'])
                coverage.append(dict(source=path,experiment=name,window=window,nominal=nominal,n=len(sub),target_months=len(months),
                    coverage=float(hit[mask].mean()),coverage_ci_time_block_low=lo,coverage_ci_time_block_high=hi,
                    nominal_within_descriptive_ci=bool(lo<=nominal<=hi) if np.isfinite(lo) else None,
                    p_value=np.nan,status='coverage diagnostic; no reliable iid binomial significance; few time blocks',confirmatory=False))
            catalog.append(dict(source=path,experiment=name,status='clustered coverage diagnostics recorded; formal p-value unavailable'))
    pd.DataFrame(coverage).to_csv(OUT/'coverage_diagnostics.csv',index=False,encoding='utf-8-sig')
    # Record every saved source including forecasts without observed labels.
    artifacts=set((ROOT/'results').glob('*/*predictions*.parquet'))|set((ROOT/'results').glob('*/*forecasts*.parquet'))
    for path in sorted(artifacts):
        relative=path.relative_to(ROOT).as_posix()
        if relative in processed:continue
        schema=pq.read_schema(path).names
        reason='no observed evaluation labels' if 'actual' not in schema else 'duplicate audit artifact; see canonical paired comparisons'
        if 'model' in schema:
            for name in pd.read_parquet(path,columns=['model']).model.unique():
                catalog.append(dict(source=relative,experiment=name,status=reason))
        else:catalog.append(dict(source=relative,experiment='all saved models',status=reason))
    grid=ROOT/'results/hypotheses/ridge_grid_metrics.csv'
    for name,g in pd.read_csv(grid).groupby('model'):
        catalog.append(dict(source=grid.relative_to(ROOT).as_posix(),experiment=name,
            status='individual predictions recovered and metrics reproduced; see results/ridge_grid' if (ROOT/'results/ridge_grid/predictions.parquet').exists()
                   else 'aggregate metrics only; paired predictions absent; significance unavailable'))
    for folder in ['architecture_probe','detector_validation','detectors_matched']:
        catalog.append(dict(source='results/'+folder,experiment='all saved scenarios',
            status='synthetic or diagnostic experiment; separate sequence-level statistics, not real forecast MAE'))
    pd.DataFrame(catalog).to_csv(OUT/'experiment_catalog.csv',index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(dict(unique_point_comparisons=len(table),comparison_aliases=len(aliases),
        catalog_records=len(catalog),coverage_diagnostics=len(coverage),holm_family_size=int(table.p_time_block.notna().sum()),
        significant_improvements_time_block_holm=int(table.significant_improvement_time_block_holm_005.sum()),
        all_results_exploratory=True,seed=cfg['seed']),indent=2))
    manifests=[]
    for path in sorted({ROOT/p for p in processed if (ROOT/p).is_file()}):
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
        manifests.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest.hexdigest(),rows=pq.ParquetFile(path).metadata.num_rows))
    (OUT/'source_manifest.json').write_text(json.dumps(manifests,indent=2))
    print('Registered',len(table),'unique paired comparisons;',len(catalog),'catalog entries',flush=True)


if __name__=='__main__':main()
