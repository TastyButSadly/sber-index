"""Evidence catalogue for candidate event features, without fitted effect sizes."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/external/events';OUT.mkdir(parents=True,exist_ok=True)


def event(id_,name,effective,published,url,mechanism,exposure,kind='step',magnitude=None,scope='Russia'):
    return dict(id=id_,name=name,effective_date=effective,source_publication_date=published,
        source_url=url,magnitude=magnitude,scope=scope,shape=kind,
        mechanism_hypothesis=mechanism,required_exposure=exposure,effect_estimated=False,
        historical_snapshot_verified=False,
        availability_rule='Source publication is a conservative knowledge bound; allow only on the following day. Undated sources excluded. Confirm actual forecast decision timestamp separately.')


def main():
    rows=[
      event('tax_reporting_2023','Единые сроки отчётности 25-го и уплаты отдельных налогов 28-го','2023-01-01','2022-09-08',
        'https://www.nalog.gov.ru/rn27/news/tax_doc_news/12463637/',
        'Обязательные платежи могут влиять на ликвидность предпринимателей; сама сдача отчётности не уменьшает доход жителей автоматически.',
        'Доля предпринимателей; фактические платежи; квартальные сроки; переносы нерабочих дней','recurring'),
      event('pension_2024','Индексация страховых пенсий неработающих на 7,5%','2024-01-01','2023-12-28',
        'https://sfr.gov.ru/branches/sakhalin/news~2023/12/28/258852',
        'Рост дохода получателей; эффект на потребление может отличаться по возрасту и категориям.',
        'Доля неработающих получателей страховой пенсии',magnitude=.075),
      event('deposit_tax_first_payment_2024','Первый платёж налога с процентов по вкладам за 2023 год','2024-12-02','2024-02-02',
        'https://www.nalog.gov.ru/rn30/news/activities_fts/14363543/',
        'Возможный отток средств перед сроком оплаты, включая ноябрь; налоговые переводы не считать потребительскими расходами.',
        'Налогооблагаемые проценты/депозиты, доля состоятельных жителей','deadline'),
      event('credit_limits_q3_2024','Макропруденциальные лимиты потребкредитования на III квартал 2024','2024-07-01','2024-05-24',
        'https://www.cbr.ru/rbr/dir_decisions/rsd_2024-05-24_35_01/',
        'Изменение доступности новых кредитов; эффект может проявляться с задержкой, независимо от уже включённой ключевой ставки.',
        'Новые выдачи потребкредитов, долговая нагрузка и банковская структура'),
      event('credit_buffers_sep_2024','Повышение надбавок по необеспеченным потребкредитам','2024-09-01','2024-06-28',
        'https://www.cbr.ru/press/pr/?file=638551914026313563FINSTAB.htm',
        'Ограничение кредитного предложения; кандидат для распределённых лагов 0–3 месяца.',
        'Региональные выдачи необеспеченных кредитов и задолженность'),
      event('mortgage_end_2024','Завершение широкой программы льготной ипотеки','2024-07-01','2024-07-26',
        'https://www.cbr.ru/press/event/?id=18869',
        'Предварительный спрос до завершения и спад после; возможное влияние на сопутствующие покупки.',
        'Ипотечные выдачи и ввод жилья; источник выбран консервативно после события, до июля нужен более ранний документ'),
      event('pension_prepaid_tomsk_2023','Томская область: часть январских пенсий и пособий перечислена в декабре','2023-12-28','2023-12-25',
        'https://sfr.gov.ru/branches/tomsk/news~2023/12/25/258543',
        'Перенос денежного потока через границу года; нельзя считать полностью новым событием только 2024 года.',
        'Число получателей с соответствующим графиком, способ доставки','cashflow_transfer',scope='Tomsk region'),
      event('pension_prepaid_2024','Январские пенсии части банковских получателей выплачены до 28 декабря','2024-12-28','2024-12-23',
        'https://sfr.gov.ru/press_center/news//~2024/12/23/269094',
        'Декабрь получает часть январского потока; в январе эти получатели не получают повторную выплату. Не эквивалентно росту годового дохода.',
        'Число получателей с обычной датой 1–9 января и банковской доставкой','cashflow_transfer'),
      event('pension_index_2025_initial','Первоначальная индексация страховых пенсий на 7,3%','2025-01-01','2024-12-23',
        'https://sfr.gov.ru/press_center/news//~2024/12/23/269094',
        'Увеличение пенсионного дохода; не умножать все расходы жителей на 1,073.',
        'Доля получателей пенсии и сумма выплат',magnitude=.073),
      event('pension_index_2025_revision','Февральская доиндексация до 9,5% и доплата за январь','2025-02-01','2025-01-24',
        'https://sfr.gov.ru/press_center/news/~2025/01/24/269618',
        'Пересмотр уровня плюс разовая доплата; 9,5% нельзя использовать в прогнозе из декабря 2024.',
        'Получатели, уже полученные январские суммы; различать уровень и разовую доплату','revision',magnitude=.095),
      event('mrot_2025','МРОТ 22 440 рублей с января 2025','2025-01-01','2024-12-17',
        'https://www.nalog.gov.ru/rn91/ifns/ifns2/info/15522777/',
        'Повышение нижней границы зарплат; передача в расходы зависит от доли затронутых работников.',
        'Доля работников около МРОТ, региональные минимумы и коэффициенты',magnitude=22440),
      event('ndfl_2025','Новая прогрессивная шкала НДФЛ с января 2025','2025-01-01','2024-12-20',
        'https://www.nalog.gov.ru/rn28/news/activities_fts/15540345/',
        'Разный эффект по распределению годовых доходов и накоплению базы в течение года; не универсальное снижение дохода на 9%.',
        'Распределение доходов по порогам, премии, исключения и вычеты'),
      event('vat_usn_2025','НДС для части плательщиков УСН с доходом выше 60 млн','2025-01-01','2025-01-08',
        'https://www.nalog.gov.ru/rn77/news/activities_fts/15560407/',
        'Возможное изменение цен и маржи продавцов; опубликованное разъяснение январское, для декабря требуется закон/методические рекомендации до origin.',
        'Доля затронутых продавцов, выбранный режим НДС, отрасли, перенос налога в цены'),
    ]
    catalogue=dict(checked_at=datetime.now(timezone.utc).isoformat(),purpose='candidate event research; not a trained model',
        events=rows,statistical_status='No new effect test: catalogue alone has no paired outcomes.',
        forecast_decision_time='Must be supplied explicitly. Dataset first-of-month labels are not publication timestamps.',
        not_implemented=['Regional utility tariffs and actual bills','Complete production calendars and bank transfer schedules',
            'Municipal age and benefit-recipient exposures','Retail sale dates and order/payment/return timing',
            'Regional emergency assistance and reimbursement dates','Cashless-payment coverage/methodology changes'])
    (OUT/'catalog.json').write_text(json.dumps(catalogue,ensure_ascii=False,indent=2),encoding='utf-8')
    # This is a deterministic knowledge audit, not a forecasting experiment.
    import pandas as pd
    grid=[]
    for asof in ['2024-09-30','2024-12-01','2024-12-31','2025-01-31']:
        for r in rows:
            known=pd.Timestamp(r['source_publication_date'])+pd.Timedelta(days=1)<=pd.Timestamp(asof)
            grid.append(dict(asof=asof,event=r['id'],known_by_conservative_source=bool(known),
                effective_date=r['effective_date'],source_publication_date=r['source_publication_date']))
    pd.DataFrame(grid).to_csv(OUT/'knowledge_audit.csv',index=False)
    checks=dict(events=len(rows),status='passed',independent_validation=False,
        revision_2025_unavailable_in_december=not any(g['known_by_conservative_source'] for g in grid
            if g['event']=='pension_index_2025_revision' and g['asof'].startswith('2024')))
    assert checks['revision_2025_unavailable_in_december']
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':main()
