"""Compare causal detectors at matched control false-alarm probability."""
import json
import numpy as np
import pandas as pd
from backtest import ROOT,load_panel

OUT=ROOT/'results/detectors_matched'


def statistics(series,method):
    mean=np.mean(series[:12]);scale=max(np.std(series[:12],ddof=1),.01)
    z=(series-mean)/scale
    upper=lower=fast=slow=running=ph=low=high=0.
    scores=[]
    for t in range(12,len(series)):
        v=z[t]
        if method=='cusum':
            upper=max(0,upper+v-.5);lower=max(0,lower-v-.5);score=max(upper,lower)
        elif method=='page_hinkley':
            running+=(v-running)/(t-11);ph+=v-running;low=min(low,ph);high=max(high,ph);score=max(ph-low,high-ph)
        elif method=='ewma':
            fast=.4*v+.6*fast;slow=.05*v+.95*slow;score=abs(fast-slow)
        else:raise ValueError(method)
        scores.append(score)
    return np.asarray(scores)


def sequences(rng,n,length):
    noise=rng.normal(0,.04,(n,length))
    for t in range(1,length):noise[:,t]+=.3*noise[:,t-1]
    return noise


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/detector_matched.json').read_text())
    controls=sequences(np.random.default_rng(cfg['seed_calibration']),cfg['calibration_controls'],cfg['length'])
    thresholds={m:float(np.quantile([statistics(row,m).max() for row in controls],.95,method='higher')) for m in cfg['methods']}
    rng=np.random.default_rng(cfg['seed_evaluation']);rows=[];sequence_results=[]
    n=cfg['evaluation_sequences_per_scenario'];tau=cfg['event_index']
    for scenario in ['control','persistent_step','gradual_change','single_outlier']:
        series=sequences(rng,n,cfg['length']);direction=np.where(np.arange(n)%2,1,-1)
        if scenario=='persistent_step':series[:,tau:]+=direction[:,None]*.2
        if scenario=='gradual_change':series[:,tau:]+=direction[:,None]*.015*np.arange(1,cfg['length']-tau+1)
        if scenario=='single_outlier':series[:,tau]+=direction*.3
        for m in cfg['methods']:
            pre=[];any_alarm=[];hits=[];delays=[]
            for sequence_id,row in enumerate(series):
                score=statistics(row,m);alarm=np.flatnonzero(score>thresholds[m])+12
                pre.append(bool(np.any(alarm<tau)));any_alarm.append(bool(len(alarm)))
                # Detection counts only sequences with no earlier false alert.
                hit=alarm[(alarm>=tau)&(alarm<tau+cfg['hit_window_months'])]
                valid=not pre[-1] and bool(len(hit));hits.append(valid)
                if valid:delays.append(int(hit[0]-tau))
                sequence_results.append(dict(method=m,scenario=scenario,sequence_id=sequence_id,
                    pre_alarm=pre[-1],any_alarm=any_alarm[-1],hit=valid if scenario!='control' else None,
                    delay=int(hit[0]-tau) if valid and scenario!='control' else None))
            rows.append(dict(method=m,scenario=scenario,n=n,threshold=thresholds[m],pre_event_alarm_fraction=float(np.mean(pre)),
                whole_sequence_alarm_fraction=float(np.mean(any_alarm)),detected_within_6_without_pre_alarm=float(np.mean(hits)) if scenario!='control' else None,
                mean_delay_detected_months=float(np.mean(delays)) if delays and scenario!='control' else None))
    pd.DataFrame(rows).to_csv(OUT/'synthetic_metrics.csv',index=False)
    pd.DataFrame(sequence_results).to_parquet(OUT/'synthetic_sequence_results.parquet',index=False)
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    local=np.log(values[:,:,target]);local-=np.median(local,axis=0)[None,:]
    signals=[]
    for m in cfg['methods']:
        for i,tid in enumerate(ids):
            scores=statistics(local[i],m)
            for t,s in enumerate(scores,start=12):signals.append(dict(territory_id=tid,observed_month=dates[t],method=m,score=s,threshold=thresholds[m],alert=bool(s>thresholds[m])))
    signals=pd.DataFrame(signals);signals.to_parquet(OUT/'real_signals.parquet',index=False)
    geo=pd.read_parquet(ROOT/'data/processed/municipalities_2024.parquet').drop_duplicates('territory_id')
    records=json.loads((ROOT/'data/external/news/registry.json').read_text(encoding='utf8'))
    risk=[]
    for o in dates:
        cutoff=(o+pd.offsets.MonthEnd(0)+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)).tz_localize('Europe/Moscow')
        for record in records:
            pub=pd.Timestamp(record['published_at'])
            if pub>cutoff or not record['event_type'].startswith('flood'):continue
            age=(cutoff-pub).days
            if age>60:continue
            affected=record.get('territory_ids')
            if affected is None:affected=geo.loc[geo.region_code==record['region_code'],'territory_id'].tolist()
            for tid in affected:
                if tid in ids:risk.append(dict(territory_id=tid,origin=o,event_id=record['event_id'],published_at=pub,
                    as_of=cutoff,event_type=record['event_type'],days_since_publication=age,risk_score=float(np.exp(-age/30)),
                    scope=record['scope'],source_url=record['url']))
    risks=pd.DataFrame(risk);risks.to_parquet(OUT/'news_risk.parquet',index=False)
    assert (risks.published_at<=risks.as_of).all()
    cases=[]
    for tid in [1673,1665,1333]:
        i=np.flatnonzero(ids==tid)[0]
        for t in range(12,len(dates)):
            if not pd.Timestamp('2024-03-01')<=dates[t]<=pd.Timestamp('2024-06-01'):continue
            row=dict(territory_id=tid,municipality=geo.set_index('territory_id').loc[tid,'municipal_district_name_short'],month=dates[t],
                actual_rub=values[i,t,target],yoy_percent=100*(values[i,t,target]/values[i,t-12,target]-1),
                local_residual=local[i,t],local_change=local[i,t]-local[i,t-1],
                available_news_count=int(((risks.territory_id==tid)&(risks.origin==dates[t])).sum()))
            for m in cfg['methods']:
                ss=signals[(signals.territory_id==tid)&(signals.observed_month==dates[t])&(signals.method==m)]
                row[m+'_alert']=bool(ss.alert.iloc[0])
            cases.append(row)
    pd.DataFrame(cases).to_csv(OUT/'real_cases.csv',index=False,encoding='utf-8-sig')
    # Unknown later observations cannot change an earlier detector statistic.
    bad=local[0].copy();bad[18:]+=100
    for m in cfg['methods']:assert np.array_equal(statistics(local[0],m)[:6],statistics(bad,m)[:6])
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    (OUT/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
    (OUT/'checks.json').write_text(json.dumps(dict(status='passed',future_observations_excluded=True,news_published_before_asof=True,
        threshold_calibration_seed_separate=True,real_event_precision_recall='not estimated; disasters are not labelled spending breaks'),indent=2))
    print(pd.DataFrame(rows).round(3).to_string(index=False),flush=True)


if __name__=='__main__':main()
