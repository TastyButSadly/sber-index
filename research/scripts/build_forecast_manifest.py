"""Separate forecast keys, evaluation truth and real-time availability status."""
import json
import pandas as pd
from backtest import ROOT

def available_external(frame,as_of):
    if 'available_at' not in frame or frame.available_at.isna().any():
        raise ValueError('External inputs require verified available_at timestamps')
    published=pd.to_datetime(frame.available_at,utc=True)
    cutoff=pd.Timestamp(as_of)
    cutoff=cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
    return frame[published<=cutoff].copy()

def main():
    p=pd.read_parquet(ROOT/'results/reference_audit/predictions.parquet',filters=[('model','==','public_ens3_mean')])
    keys=['territory_id','origin','date','horizon']
    assert not p.duplicated(keys).any()
    manifest=p[keys].rename(columns={'date':'target_month'}).copy()
    manifest['last_observed_month']=manifest.origin
    manifest['observation_time_as_of']=manifest.origin+pd.offsets.MonthEnd(0)
    manifest['verified_real_time_as_of']=pd.NaT
    manifest['availability_status']='observation-time experiment; release timestamps unverified'
    manifest.to_parquet(ROOT/'data/processed/forecast_manifest.parquet',index=False)
    truth=p[keys+['actual','seasonal_value']].rename(columns={'date':'target_month'})
    truth['verified_target_available_at']=pd.NaT
    truth.to_parquet(ROOT/'data/processed/evaluation_truth.parquet',index=False)
    contract={'join_keys':['territory_id or documented region crosswalk','observation_month'],
        'required_external_fields':['source_url','observation_date','available_at','vintage','value'],
        'rule':'available_at <= as_of; forecast covariates also require issued_at <= as_of',
        'calendar':'known-in-advance values with calendar version provenance',
        'missing_release_date':'exclude from real-time experiment; do not invent availability',
        'truth':'evaluation_truth.parquet is for matured labels and scoring only',
        'news':'published_at/first_seen separate from event_date and effective_date',
        'weather':'future realized weather excluded; archived forecast issue time required',
        'point_features':'factor/local/calendars maintained independently of uncertainty'}
    (ROOT/'configs/data_availability_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf8')
    print('Saved',len(manifest),'forecast keys and separate evaluation truth.')

if __name__=='__main__':main()
