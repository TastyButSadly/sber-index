"""Select cases and write explanations before news lookup; audit mobility eligibility."""
import json
import hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from backtest import ROOT,load_panel
from final_protocol_intervals import scale_at
from final_detection_benchmark import bh,pvalues

OUT=ROOT/'results/final_project'


def main():
    cfg=json.loads((ROOT/'configs/final_project.json').read_text())
    ids,dates,cats,values=load_panel();total=cats.index('Все категории')
    allp=pd.read_parquet(OUT/'predictions.parquet');base=allp[allp.model==cfg['point_model']]
    intervals=pd.read_parquet(OUT/'interval_predictions.parquet',columns=['territory_id','origin','date','horizon','actual','prediction','interval_method','nominal','lower','upper','scale_log','q'])
    cases=intervals[(intervals.interval_method=='adaptive_log')&(intervals.nominal==.9)&(intervals.horizon==1)].copy()
    cases['signed_log_error']=np.log(cases.actual/cases.prediction)
    cases['common_log_surprise']=cases.groupby('date').signed_log_error.transform('median')
    cases['local_log_surprise']=cases.signed_log_error-cases.common_log_surprise
    cases['local_standardized_surprise']=cases.local_log_surprise/cases.scale_log
    geo=pd.read_parquet(ROOT/'data/processed/municipalities_2024.parquet').set_index('territory_id')
    cases['region_code']=cases.territory_id.map(geo.region_code)
    cases['municipality']=cases.territory_id.map(geo.municipal_district_name_short)
    cases['region']=cases.territory_id.map(geo.region_name)
    peers=[]
    for origin,g in cases.groupby('origin'):
        scales,cal,_=scale_at(origin,base,cfg,ids,dates,values,total)
        cal=cal[cal.horizon==1].copy();cal['e']=np.log(cal.actual/cal.prediction)
        cal['e']-=cal.groupby('date').e.transform('median')
        scores=np.sort(abs(cal.e.to_numpy())/scales[1].reindex(cal.territory_id).to_numpy())
        pv=pvalues(abs(g.local_standardized_surprise.to_numpy()),scores)
        cases.loc[g.index,'conformal_p']=pv;cases.loc[g.index,'bh_alert_q005']=bh(pv,.05)
        o=dates.get_loc(origin);log=np.log(values[:,:o+1,total]);local=log-np.median(log,axis=0)[None,:]
        feature=np.c_[log[:,-1],values[:,o,cats.index('Маркетплейсы')]/values[:,o,total],
                      values[:,o,cats.index('Продовольствие')]/values[:,o,total],np.std(np.diff(local[:,-7:],axis=1),axis=1)]
        feature=(feature-feature.mean(axis=0))/np.maximum(feature.std(axis=0),1e-6)
        observed=g.set_index('territory_id');region=geo.region_code.reindex(ids).to_numpy()
        for row in g.itertuples():
            i=np.flatnonzero(ids==row.territory_id)[0];candidates=np.flatnonzero((region==region[i])&(ids!=ids[i]))
            if len(candidates)<5:candidates=np.flatnonzero(ids!=ids[i])
            nearest=candidates[np.argsort(np.sum((feature[candidates]-feature[i])**2,axis=1))[:5]]
            peer_ids=ids[nearest];peer_error=float(observed.local_log_surprise.reindex(peer_ids).median())
            cases.loc[row.Index,'peer_standardized_surprise']=(row.local_log_surprise-peer_error)/row.scale_log
            for tid in peer_ids:peers.append(dict(territory_id=row.territory_id,origin=origin,peer_id=tid,selection='5 nearest past-state peers, same region when possible'))
    selections=[];used=set()
    for rule in cfg['case_rules']:
        eligible=cases[~cases.territory_id.isin(used)]
        if rule=='negative_surprise':index=eligible.local_standardized_surprise.idxmin()
        elif rule=='positive_surprise':index=eligible.local_standardized_surprise.idxmax()
        elif rule=='national_movement_low_local_surprise':
            target=eligible.groupby('date').common_log_surprise.first().abs().idxmax()
            pool=eligible[(eligible.date==target)&(~eligible.bh_alert_q005.astype(bool))]
            if pool.empty:pool=eligible[eligible.date==target]
            index=pool.local_standardized_surprise.abs().idxmin()
        else:index=eligible.peer_standardized_surprise.abs().idxmax()
        row=cases.loc[index].to_dict();row['case_rule']=rule;used.add(row['territory_id']);selections.append(row)
    selected=pd.DataFrame(selections)
    selected.to_csv(OUT/'selected_cases.csv',index=False,encoding='utf-8-sig')
    selection=dict(selected_at=datetime.now(timezone.utc).isoformat(),rules=cfg['case_rules'],
        source_sha256=hashlib.sha256((OUT/'interval_predictions.parquet').read_bytes()).hexdigest(),
        news_read_before_selection=False,external_event_explanations='not yet looked up; this file freezes choices',
        cases=[dict(rule=r.case_rule,territory_id=int(r.territory_id),origin=str(r.origin.date()),target=str(r.date.date()),municipality=r.municipality,region=r.region) for r in selected.itertuples()])
    (OUT/'case_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    peers=pd.DataFrame(peers);peers=peers.merge(selected[['territory_id','origin']],on=['territory_id','origin'])
    peer_history=peers.merge(cases[['territory_id','origin','actual','prediction','lower','upper','local_standardized_surprise']],
                             left_on=['peer_id','origin'],right_on=['territory_id','origin'],suffixes=('','_peer'))
    peer_history.to_csv(OUT/'case_peers.csv',index=False)
    category_rows=[]
    for row in selected.itertuples():
        i=np.flatnonzero(ids==row.territory_id)[0];t=dates.get_loc(row.date);o=dates.get_loc(row.origin)
        other=[j for j in range(len(cats)) if j!=total]
        prior=values[i,t-12,total];actual=values[i,t,total]
        remainder=0.
        for j in other:
            difference=values[i,t,j]-values[i,t-12,j];remainder+=difference
            category_rows.append(dict(case_rule=row.case_rule,territory_id=row.territory_id,target=row.date,category=cats[j],
                actual_rub=values[i,t,j],year_ago_rub=values[i,t-12,j],change_yoy_rub=difference,
                interpretation='observed category YoY contribution; not a causal decomposition of ensemble prediction error'))
        category_rows.append(dict(case_rule=row.case_rule,territory_id=row.territory_id,target=row.date,category='Other categories',
            actual_rub=actual-values[i,t,other].sum(),year_ago_rub=prior-values[i,t-12,other].sum(),change_yoy_rub=actual-prior-remainder,
            interpretation='total minus named categories'))
    pd.DataFrame(category_rows).to_csv(OUT/'case_categories.csv',index=False,encoding='utf-8-sig')
    cases.to_parquet(OUT/'real_anomaly_scores.parquet',index=False)
    mobility=ROOT/'data/raw/mobility_current.parquet';m=pd.read_parquet(mobility)
    audit=dict(checked_at=datetime.now(timezone.utc).isoformat(),source='https://sberindex.ru/api/dataset/v1/download/indeks-mobilnosti/parquet',
        file=str(mobility.relative_to(ROOT)),sha256=hashlib.sha256(mobility.read_bytes()).hexdigest(),rows=len(m),
        periods=m.groupby('period').size().to_dict(),frequency_values=m.freq.unique().tolist(),
        geographic_field='ref_area display names; no territory_id',unique_names=int(m.ref_area.nunique()),
        available_at_present=False,monthly_history_before_2024_origin=False,
        hypothesis=cfg['mobility_hypothesis'],ablation_status='not estimable on this source: only 2024-12-31 and 2025-11-01, no monthly within-2024 changes, unverified geography/release times',
        p_value=None,interpretation='data eligibility failure, not a measured negative predictive result; no further external source search')
    (OUT/'mobility_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    assert len(selected)==4 and selected.territory_id.nunique()==4
    check=pd.DataFrame(category_rows).groupby('territory_id').change_yoy_rub.sum()
    for row in selected.itertuples():
        i=np.flatnonzero(ids==row.territory_id)[0];t=dates.get_loc(row.date)
        assert np.isclose(check.loc[row.territory_id],values[i,t,total]-values[i,t-12,total])
    (OUT/'case_checks.json').write_text(json.dumps(dict(status='passed',selection_frozen_before_news=True,
        peers_use_only_origin_features=True,category_yoy_contributions_add_up=True,
        mobility_ablation_not_fabricated=True,real_event_precision_recall_unavailable=True),indent=2))
    print(selected[['case_rule','municipality','region','date','local_standardized_surprise','bh_alert_q005']].to_string(index=False),flush=True)


if __name__=='__main__':main()
