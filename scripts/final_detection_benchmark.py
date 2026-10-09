"""One real-background semi-synthetic benchmark, calibrated alerts per MO-month."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from backtest import ROOT,load_panel
from experiment_statistics import sign_flip,holm

OUT=ROOT/'results/final_project/detection'


def scores(z,method):
    n,length=z.shape;out=np.empty_like(z)
    upper=np.zeros(n);lower=upper.copy();running=upper.copy();cumulative=upper.copy();minimum=upper.copy();maximum=upper.copy();fast=upper.copy()
    for t in range(length):
        v=z[:,t]
        if method=='cusum':
            upper=np.maximum(0,upper+v-.5);lower=np.maximum(0,lower-v-.5);s=np.maximum(upper,lower)
        elif method=='page_hinkley':
            running+=(v-running)/(t+1);cumulative+=v-running
            minimum=np.minimum(minimum,cumulative);maximum=np.maximum(maximum,cumulative)
            s=np.maximum(cumulative-minimum,maximum-cumulative)
        elif method=='ewma':fast=.4*v+.6*fast;s=abs(fast)
        else:s=abs(v)
        out[:,t]=s
    return out


def bh(p,q):
    n=len(p);ordered=np.sort(p);ok=ordered<=q*np.arange(1,n+1)/n
    cutoff=ordered[np.flatnonzero(ok)[-1]] if ok.any() else -1.
    return p<=cutoff


def pvalues(score,reference):
    return (1+len(reference)-np.searchsorted(reference,score,side='left'))/(len(reference)+1)


def notify(raw,cooldown):
    last=np.full(raw.shape[0],-cooldown,dtype=int);alarms=np.zeros_like(raw,dtype=bool)
    for t in range(raw.shape[1]):
        yes=raw[:,t]&(t-last>=cooldown);alarms[:,t]=yes;last[yes]=t
    return alarms


def alarms(score,method,threshold,cfg,reference):
    raw=score>threshold if not method.startswith('conformal_bh') else np.column_stack([bh(pvalues(score[:,t],reference),threshold) for t in range(score.shape[1])])
    return notify(raw,cfg['notification_refractory_months'])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    full=json.loads((ROOT/'configs/final_project.json').read_text());cfg=full['benchmark']
    (OUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
    p=pd.read_parquet(ROOT/'results/final_project/predictions.parquet',columns=['model','horizon','territory_id','date','actual','prediction'])
    p=p[(p.model==full['point_model'])&(p.horizon==1)]
    ids,dates,cats,values=load_panel();total=cats.index('Все категории')
    target_dates=sorted(p.date.unique());error=np.log(p.pivot(index='territory_id',columns='date',values='actual').reindex(ids).to_numpy()/
                                                     p.pivot(index='territory_id',columns='date',values='prediction').reindex(ids).to_numpy())
    interval=pd.read_parquet(ROOT/'results/final_project/interval_predictions.parquet',
        columns=['interval_method','nominal','horizon','territory_id','date','scale_log'])
    interval=interval[(interval.interval_method=='adaptive_log')&(interval.nominal==.9)&(interval.horizon==1)]
    scale=interval.pivot(index='territory_id',columns='date',values='scale_log').reindex(index=ids,columns=target_dates).to_numpy()
    log=np.log(values[:,:12,total]);local=log-np.median(log,axis=0)[None,:]
    fallback=np.maximum(np.median(abs(np.diff(local,axis=1)),axis=1),.005)
    scale=np.where(np.isfinite(scale),scale,fallback[:,None]);length=error.shape[1]
    assert cfg['event_index']+cfg['hit_window_months']<=length
    background=pd.DataFrame(error,index=ids,columns=target_dates);background.to_parquet(OUT/'real_log_errors.parquet')
    def panel(rng,scenario):
        ix=rng.integers(0,len(ids),cfg['municipalities_per_panel']);e=error[ix].copy();s=scale[ix]
        affected=np.zeros(len(ix),dtype=bool);delta=np.zeros_like(e);tau=cfg['event_index']
        if scenario!='control':
            selected=np.arange(len(ix)) if scenario=='national' else rng.choice(len(ix),int(len(ix)*cfg['affected_fraction']),replace=False)
            affected[selected]=True;direction=rng.choice([-1.,1.],len(selected))
            if scenario=='step':delta[selected,tau:]=direction[:,None]*cfg['log_step']
            elif scenario=='drift':delta[selected,tau:]=direction[:,None]*cfg['log_drift_per_month']*np.arange(1,length-tau+1)
            elif scenario=='temporary':delta[selected,tau:tau+cfg['temporary_duration_months']]=direction[:,None]*cfg['log_temporary_shock']
            else:delta[:,tau:]=cfg['log_step']
        e+=delta
        # A local signal is a deviation from the contemporaneous national movement.
        z=(e-np.median(e,axis=0)[None,:])/s
        return z,affected,ix
    rng=np.random.default_rng(cfg['reference_seed']);reference=np.sort(np.concatenate([abs(panel(rng,'control')[0]).ravel() for _ in range(4)]))
    rng=np.random.default_rng(cfg['calibration_seed']);calibration=[panel(rng,'control')[0] for _ in range(cfg['calibration_panels'])]
    thresholds={};calibration_rows=[]
    for method in cfg['methods']:
        bank=[scores(z,method) for z in calibration]
        grid=np.r_[np.linspace(.001,.99,35),1.] if method=='conformal_bh' else np.quantile(np.concatenate(bank),np.linspace(.7,.9999,80))
        candidates=[]
        for threshold in grid:
            rate=float(np.mean([alarms(s,method,threshold,cfg,reference).mean() for s in bank]))
            candidates.append((float(threshold),rate))
        valid=[x for x in candidates if x[1]<=cfg['target_false_alerts_per_mo_month']]
        threshold,rate=min(valid,key=lambda a:cfg['target_false_alerts_per_mo_month']-a[1])
        if method=='conformal_bh':
            # Refine numerical matching of the specified null alert budget, using calibration panels only.
            upper=min([x[0] for x in candidates if x[0]>threshold and x[1]>cfg['target_false_alerts_per_mo_month']],default=1.)
            refined=[]
            for q in np.linspace(threshold,upper,101):
                r=float(np.mean([alarms(s,method,q,cfg,reference).mean() for s in bank]))
                if r<=cfg['target_false_alerts_per_mo_month']:refined.append((float(q),r))
            threshold,rate=min(refined,key=lambda a:cfg['target_false_alerts_per_mo_month']-a[1])
        thresholds[method]=threshold;calibration_rows.append(dict(method=method,threshold_or_bh_q=threshold,calibration_alerts_per_100_mo_months=100*rate,
            unit='BH nominal q' if method=='conformal_bh' else 'normalized detector threshold'))
        print('Calibrated detector',method,threshold,rate,flush=True)
    pd.DataFrame(calibration_rows).to_csv(OUT/'calibration.csv',index=False)
    # Explicit diagnostic at conventional nominal FDR q=.05; not a matched-FPR competitor.
    thresholds['conformal_bh_q005']=.05
    evaluation_methods=cfg['methods']+['conformal_bh_q005']
    writer=None;panels=[];rng=np.random.default_rng(cfg['evaluation_seed'])
    for scenario in cfg['scenarios']:
        for panel_id in range(cfg['evaluation_panels']):
            z,affected,ix=panel(rng,scenario);tau=cfg['event_index'];end=tau+cfg['hit_window_months']
            for method in evaluation_methods:
                alarm=alarms(scores(z,method),method,thresholds[method],cfg,reference)
                before=alarm[:,:tau].any(axis=1);event_hit=alarm[:,tau:end].any(axis=1)&~before
                delay=np.argmax(alarm[:,tau:end],axis=1).astype(float);delay[~event_hit]=np.nan
                # For temporary events, all non-injected months count as null; persistent and drift continue to end.
                changed=np.zeros_like(alarm,dtype=bool)
                stop=tau+cfg['temporary_duration_months'] if scenario=='temporary' else length
                if scenario not in ['control','national']:changed[affected,tau:stop]=True
                false=int((alarm&~changed).sum());null=int((~changed).sum());alerts=int(alarm.sum())
                panels.append(dict(scenario=scenario,panel_id=panel_id,method=method,
                    recall=float(event_hit[affected].mean()) if scenario not in ['control','national'] else np.nan,
                    false_alarm_rate=false/null,realized_false_discovery_fraction=false/alerts if alerts else 0.,
                    alerts=alerts,false_alerts=false,null_mo_months=null,pre_alarm_fraction=float(before.mean()),
                    mean_delay_detected=float(np.nanmean(delay[affected])) if scenario not in ['control','national'] and affected.any() and np.isfinite(delay[affected]).any() else np.nan,
                    capped_delay=float(np.where(event_hit[affected],delay[affected],cfg['hit_window_months']).mean()) if scenario not in ['control','national'] else np.nan))
                outcome=pd.DataFrame(dict(scenario=scenario,panel_id=panel_id,sequence_id=np.arange(len(ix)),source_territory_id=ids[ix],method=method,
                    affected=affected,pre_alarm=before,hit=event_hit&affected,delay=delay,
                    false_alarms=(alarm&~changed).sum(axis=1),null_months=(~changed).sum(axis=1)))
                table=pa.Table.from_pandas(outcome,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(OUT/'sequence_results.parquet',table.schema)
                writer.write_table(table)
        print('Evaluated scenario',scenario,flush=True)
    writer.close()
    panel_results=pd.DataFrame(panels);panel_results.to_csv(OUT/'panel_results.csv',index=False)
    summary=panel_results.groupby(['scenario','method'],as_index=False).agg(recall=('recall','mean'),false_alarm_rate=('false_alarm_rate','mean'),
        mean_delay_detected=('mean_delay_detected','mean'),capped_delay=('capped_delay','mean'),
        realized_fdr=('realized_false_discovery_fraction','mean'),mean_alerts=('alerts','mean'))
    summary.to_csv(OUT/'metrics.csv',index=False)
    # Statistical unit is an independently resampled PANEL, not 2016 dependent municipalities.
    comparisons=[];rng=np.random.default_rng(20261009)
    for scenario,g in panel_results.groupby('scenario'):
        for metric in ['false_alarm_rate']+(['recall','capped_delay'] if scenario in ['step','drift','temporary'] else []):
            wide=g.pivot(index='panel_id',columns='method',values=metric)
            for method in evaluation_methods:
                if method=='conformal_bh':continue
                delta=(wide.conformal_bh-wide[method]).to_numpy()
                ix=rng.integers(0,len(delta),(10000,len(delta)));lo,hi=np.quantile(delta[ix].mean(axis=1),[.025,.975])
                comparisons.append(dict(scenario=scenario,metric=metric,candidate='conformal_bh',reference=method,
                    panels=len(delta),delta=float(delta.mean()),ci_low=lo,ci_high=hi,p_sign_flip=sign_flip(delta),
                    scope='independent resampled panels conditional on fixed real background; simulation only'))
    comparisons=pd.DataFrame(comparisons);comparisons['p_holm']=holm(comparisons.p_sign_flip)
    comparisons.to_csv(OUT/'paired_statistics.csv',index=False)
    # National injection is removed exactly before local scoring.
    er=error.copy();er[:,cfg['event_index']:]+=cfg['log_step']
    assert np.allclose(er-np.median(er,axis=0),error-np.median(error,axis=0),atol=1e-12)
    x=np.ones((5,length));bad=x.copy();bad[:,8:]=100
    for m in cfg['methods']:assert np.array_equal(scores(x,m)[:,:8],scores(bad,m)[:,:8])
    (OUT/'checks.json').write_text(json.dumps(dict(status='passed',independent_reference_calibration_evaluation_seeds=True,
        identical_injected_panels_across_methods=True,calibration_targets_fixed_without_evaluation_feedback=True,
        national_common_movement_removed=True,detector_future_inputs_excluded=True,
        statistical_unit='panel, conditional on fixed background',FDR_guarantee='not established under dependent nonexchangeable scores',
        event_scope='after-observation detection of injected change; no real economic event labels'),indent=2))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
