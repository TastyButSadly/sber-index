"""Region-mapped, lagged research features; revised snapshots are not as-of vintages."""
import hashlib
import json
import re
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from backtest import ROOT


def normalize(s):
    s=s.lower().replace('ё','е').replace('h','н')
    s=re.sub(r'^\s*(в том числе\s*)?(г\.\s*)?','',s)
    s=s.replace('автономный','авт').replace('автономная','авт').replace('авт.','авт')
    s=s.replace('- кузбасс','').replace('– кузбасс','')
    return re.sub(r'[^а-я]','',s)


def main():
    out=ROOT/'data/external/regional';out.mkdir(parents=True,exist_ok=True)
    geo=pd.read_parquet(ROOT/'data/processed/municipalities_2024.parquet')[['region_code','region_name']].drop_duplicates()
    sources={name:pd.DataFrame(json.loads((out/f'{name}_raw.json').read_text(encoding='utf8'))) for name in ['prices','wages']}
    histories={};crosswalk=[]
    for name,frame in sources.items():
        frame['date']=pd.to_datetime(frame.date);frame['normalized']=frame.source_region_name.map(normalize)
        mapped=[]
        for row in geo.itertuples():
            # Municipal regions exclude the separately modelled autonomous districts.
            key=normalize(row.region_name)
            if row.region_code==29:
                key=normalize('Архангельская область (кроме Ненецкого автономного округа)' if name=='prices' else 'Архангельская область без авт. округа.')
            if row.region_code==72:
                key=normalize('Тюменская область (кроме Ханты-Мансийского автономного округа - Югры и Ямало-Ненецкого автономного округа)' if name=='prices' else 'Тюменская область без авт. округов')
            sub=frame[frame.normalized==key].copy()
            if sub.empty:raise ValueError(f'Unmapped region: {name}, {row.region_code}, {row.region_name}')
            assert not sub.date.duplicated().any()
            sub['region_code']=row.region_code;mapped.append(sub)
            crosswalk.append(dict(source=name,region_code=int(row.region_code),region_name=row.region_name,
                source_region_names=sorted(sub.source_region_name.unique().tolist())))
        histories[name]=pd.concat(mapped,ignore_index=True)
    features=[]
    country=sources['wages'].set_index(['normalized','date']).wage_rub.loc[normalize('Российская Федерация')].sort_index()
    dates=pd.date_range('2023-01-01','2024-12-01',freq='MS')
    for row in geo.itertuples():
        p=histories['prices'].query('region_code==@row.region_code').set_index('date').sort_index()
        w=histories['wages'].query('region_code==@row.region_code').set_index('date').sort_index().wage_rub
        f=pd.DataFrame(index=dates)
        for category in ['all','food','nonfood','services']:
            rate=np.log(p['cpi_'+category]/100)
            f['cpi_'+category+'_mom_lag2']=rate.shift(2).reindex(dates)
            f['cpi_'+category+'_quarter_lag2']=rate.rolling(3).sum().shift(2).reindex(dates)
            f['cpi_'+category+'_yoy_lag2']=rate.rolling(12).sum().shift(2).reindex(dates)
        log=np.log(w);national=np.log(country.reindex(w.index))
        f['wage_relative_level_lag3']=(log-national).shift(3).reindex(dates)
        f['wage_yoy_lag3']=log.diff(12).shift(3).reindex(dates)
        f['wage_relative_yoy_lag3']=(log.diff(12)-national.diff(12)).shift(3).reindex(dates)
        f['wage_quarter_lag3']=log.diff(3).shift(3).reindex(dates)
        assert np.isfinite(f.loc['2023-09-01':].to_numpy()).all()
        f['region_code']=row.region_code;f.index.name='date';features.append(f.reset_index())
    pd.concat(features,ignore_index=True).to_parquet(out/'features.parquet',index=False)
    for name,frame in histories.items():frame.drop(columns='normalized').to_parquet(out/(name+'_history.parquet'),index=False)
    (out/'crosswalk.json').write_text(json.dumps(crosswalk,ensure_ascii=False,indent=2),encoding='utf8')
    manifest=dict(retrieved_at=datetime.now(timezone.utc).isoformat(),research_only=True,available_at=None,vintage='2026 snapshot',
        first_seen_at=None,tls_certificate_verified=False,
        strict_real_time_eligible=False,history_end='2024-12-01',lag_months={'prices':2,'wages':3},
        warning='revisions and exact historical publication timestamps unknown; conservative lags do not establish as-of availability',
        sources=[dict(name=name,source_url=url,published_snapshot_date=date,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
          for name,url,date,path in [('prices','https://rosstat.gov.ru/storage/mediabank/ipc_RF_fo_sub_08-2026.xlsx','2026-09-11','data/external/regional_prices.xlsx'),
                                    ('wages','https://rosstat.gov.ru/storage/mediabank/tab2-zpl_07-2026.xlsx','2026-09-30','data/external/regional_wages.xlsx')]])
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('Mapped',len(geo),'regions; 16 lagged features; 2023-09 onward complete',flush=True)


if __name__=='__main__':main()
