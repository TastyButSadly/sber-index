"""Small audited primary-source registry; published time is distinct from event time."""
import hashlib
import json
from datetime import datetime,timezone
import requests
import pandas as pd
from bs4 import BeautifulSoup
from backtest import ROOT


def main():
    out=ROOT/'data/external/news';out.mkdir(parents=True,exist_ok=True)
    seeds=[
        dict(event_id='cbr_initial16',published_at='2023-12-22T23:59:59+03:00',effective_at='2023-12-18T00:00:00+03:00',
             url='https://www.cbr.ru/rbr/dir_decisions/rsd_2023-12-15_20_02/',scope='national',event_type='policy_rate',rate=16.,
             summary='Установление ставки 16%; доступность консервативно по дате обновления страницы, не по дате заседания.',timestamp_basis='page updated 2023-12-22; conservative end of day'),
        dict(event_id='cbr_apr16',published_at='2024-04-26T13:30:00+03:00',effective_at='2024-04-26T13:30:00+03:00',
             url='https://www.cbr.ru/press/pr/?file=26042024_133000Key.htm',scope='national',event_type='policy_rate',rate=16.,summary='Сохранение ставки 16%.',timestamp_basis='press release timestamp'),
        dict(event_id='cbr_jul18',published_at='2024-07-26T13:30:00+03:00',effective_at='2024-07-29T00:00:00+03:00',
             url='https://www.cbr.ru/press/pr/?file=26072024_133000Key.htm',scope='national',event_type='policy_rate',rate=18.,summary='Объявлено повышение ставки до 18%.',timestamp_basis='press release timestamp'),
        dict(event_id='cbr_sep19',published_at='2024-09-13T13:30:00+03:00',effective_at='2024-09-16T00:00:00+03:00',
             url='https://www.cbr.ru/press/pr/?file=13092024_133000Key.htm',scope='national',event_type='policy_rate',rate=19.,summary='Объявлено повышение ставки до 19%.',timestamp_basis='press release timestamp'),
        dict(event_id='cbr_oct21',published_at='2024-10-25T13:30:00+03:00',effective_at='2024-10-28T00:00:00+03:00',
             url='https://www.cbr.ru/press/pr/?file=25102024_133000Key.htm',scope='national',event_type='policy_rate',rate=21.,summary='Объявлено повышение ставки до 21%.',timestamp_basis='press release timestamp'),
        dict(event_id='flood_apr10',published_at='2024-04-10T12:55:00+03:00',effective_at=None,
             url='https://mchs.gov.ru/deyatelnost/press-centr/novosti/5252100',scope='municipal',territory_ids=[1673,1665],event_type='flood_observed',rate=None,
             summary='Сообщение о реагировании на уже происходящий паводок в Орске и Оренбурге; не предупреждение до начала паводка.',timestamp_basis='page timestamp'),
        dict(event_id='kurgan_warning_apr22',published_at='2024-04-22T06:30:00+03:00',effective_at=None,
             url='https://45.mchs.gov.ru/deyatelnost/press-centr/operativnaya-informaciya/5259955',scope='regional',region_code=45,event_type='flood_warning',rate=None,
             summary='Предупреждение о возможном подтоплении отдельных населённых пунктов в ближайшие 1–2 дня во время уже идущего паводка; региональная экспозиция не означает затопление всех МО.',timestamp_basis='page timestamp'),
    ]
    manifest=[];registry=[]
    for seed in seeds:
        path=out/(seed['event_id']+'.html')
        extracted=out/(seed['event_id']+'.web.json')
        if extracted.exists():
            raw=extracted.read_bytes();text=raw.decode('utf8');source_format='web_extracted_json'
        else:
            if not path.exists():
                response=requests.get(seed['url'],timeout=40);response.raise_for_status();path.write_bytes(response.content)
            raw=path.read_bytes();text=BeautifulSoup(raw,'html.parser').get_text(' ',strip=True);source_format='html'
        if len(text)<200:raise ValueError('Source too short: '+seed['url'])
        (out/(seed['event_id']+'.txt')).write_text(text,encoding='utf8')
        entry=dict(seed,source_format=source_format,source_sha256=hashlib.sha256(raw).hexdigest(),retrieved_at=datetime.now(timezone.utc).isoformat(),
                   first_seen_at=None,availability_basis='trusted page publication/update timestamp; no archived first-seen evidence')
        registry.append(entry);manifest.append(dict(event_id=seed['event_id'],url=seed['url'],sha256=entry['source_sha256'],bytes=len(raw)))
        print('Saved',seed['event_id'],len(raw),flush=True)
    (out/'registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    # Monthly as-of feature ledger; no actual future policy decision enters a forecast.
    _,dates,_,_=__import__('backtest').load_panel()
    rows=[]
    for o in dates:
        cutoff=(o+pd.offsets.MonthEnd(0)+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)).tz_localize('Europe/Moscow')
        available=[r for r in registry if pd.Timestamp(r['published_at'])<=cutoff]
        rates=[r for r in available if r['event_type']=='policy_rate']
        latest=max(rates,key=lambda r:r['published_at']) if rates else None
        rows.append(dict(origin=o,as_of=cutoff,announced_policy_rate=latest['rate'] if latest else None,
            rate_source_event=latest['event_id'] if latest else None,news_count=len(available)))
    pd.DataFrame(rows).to_parquet(out/'monthly_asof.parquet',index=False)
    assert not any(r['event_id']=='cbr_jul18' for r in registry if pd.Timestamp(r['published_at'])<=pd.Timestamp('2024-06-30T23:59:59+03:00'))


if __name__=='__main__':main()
