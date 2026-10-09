"""Official macro histories with explicit conservative lags; no vintage guarantee."""
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from backtest import ROOT


def main():
    out=ROOT/'data/external/macro';out.mkdir(parents=True,exist_ok=True)
    manifest=[];series={}
    for name,code in [('usd','R01235'),('cny','R01375')]:
        url='https://www.cbr.ru/scripts/XML_dynamic.asp'
        params={'date_req1':'01/01/2018','date_req2':'31/12/2024','VAL_NM_RQ':code}
        path=out/(name+'.xml')
        if not path.exists():
            r=requests.get(url,params=params,timeout=60);r.raise_for_status();path.write_bytes(r.content)
        raw=path.read_bytes();root=ET.fromstring(raw)
        rows=[]
        for record in root.findall('Record'):
            value=float(record.findtext('Value').replace(',','.'));nominal=float(record.findtext('Nominal'))
            rows.append(dict(date=pd.to_datetime(record.attrib['Date'],dayfirst=True),value=value/nominal))
        daily=pd.DataFrame(rows).set_index('date').sort_index()
        if len(daily)<1000:raise ValueError('Insufficient official FX history')
        daily.to_csv(out/(name+'_daily.csv'))
        series[name]=daily.value.resample('MS').mean()
        manifest.append(dict(name=name,source_url=requests.Request('GET',url,params=params).prepare().url,
            sha256=hashlib.sha256(raw).hexdigest(),rows=len(daily),definition='official RUB per 1 currency unit; monthly mean',
            first_seen_at=None,publication_timestamp='unknown; effective dates supplied',lag_months=1))
    params={'UniDbQuery.Posted':'True','UniDbQuery.From':'01.01.2018','UniDbQuery.To':'31.12.2024'}
    url='https://www.cbr.ru/hd_base/infl/';path=out/'inflation.html'
    if not path.exists():
        r=requests.get(url,params=params,timeout=60);r.raise_for_status();path.write_bytes(r.content)
    raw=path.read_bytes();rows=[]
    for tr in BeautifulSoup(raw,'html.parser').select('table tr'):
        cells=[td.get_text(' ',strip=True) for td in tr.find_all('td')]
        if len(cells)<4:continue
        label=cells[0];parts=label.split('.')
        if len(parts)!=2 or not (parts[0].isdigit() and parts[1].isdigit()):continue
        date=pd.Timestamp(int(parts[1]),int(parts[0]),1)
        if pd.Timestamp('2018-01-01')<=date<=pd.Timestamp('2024-12-01'):
            numeric=[float(v.replace(',','.').replace(' ','').replace('\xa0','')) for v in cells[1:4]]
            rows.append(dict(date=date,policy_rate=numeric[0],inflation_yoy=numeric[1]))
    macro=pd.DataFrame(rows).set_index('date').sort_index()
    if len(macro)<70:raise ValueError('Inflation query did not return requested historical range')
    manifest.append(dict(name='inflation_and_rate',source_url=requests.Request('GET',url,params=params).prepare().url,
        sha256=hashlib.sha256(raw).hexdigest(),rows=len(macro),definition='CPI YoY %, policy rate at month end',
        first_seen_at=None,publication_timestamp='unknown; latest historical snapshot',inflation_lag_months=2,policy_rate_lag_months=1))
    for name,s in series.items():macro[name]=s.reindex(macro.index)
    if macro.isna().any().any():raise ValueError('Missing macro months')
    macro.to_parquet(out/'monthly_history.parquet')
    features=pd.DataFrame(index=macro.index)
    for name in ['usd','cny']:
        log=np.log(macro[name]);features[name+'_mom_lag1']=log.diff().shift(1);features[name+'_yoy_lag1']=log.diff(12).shift(1)
    features['inflation_lag2']=macro.inflation_yoy.shift(2)/100
    features['inflation_change3_lag2']=macro.inflation_yoy.diff(3).shift(2)/100
    features['rate_lag1']=macro.policy_rate.shift(1)/100
    features['rate_change3_lag1']=macro.policy_rate.diff(3).shift(1)/100
    features.to_parquet(out/'monthly_features.parquet')
    (out/'manifest.json').write_text(json.dumps(dict(retrieved_at=datetime.now(timezone.utc).isoformat(),sources=manifest,
        warning='lags are conservative modelling assumptions, not verified historical release timestamps; no archived vintages'),indent=2))
    print('Official monthly history',len(macro),'months;',macro.index.min().date(),macro.index.max().date(),flush=True)


if __name__=='__main__':main()
