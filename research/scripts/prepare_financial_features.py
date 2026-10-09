"""Read official workbooks; retain definitions and conservative lag assumptions."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/external/financial'
MONTHS = ['январь','февраль','март','апрель','май','июнь','июль','август','сентябрь','октябрь','ноябрь','декабрь']


def rate_series(name, column):
    d = pd.read_excel(OUT / f'{name}.xlsx', sheet_name=0, header=None)
    result = {}
    for row in d.itertuples(index=False, name=None):
        match = re.fullmatch(r'([а-я]+)\s+(\d{4})', str(row[0]).strip().lower())
        if match and match[1] in MONTHS:
            result[pd.Timestamp(int(match[2]), MONTHS.index(match[1])+1, 1)] = float(row[column])
    return pd.Series(result).sort_index(), str(d.iloc[4, column])


def main():
    loan, loan_label = rate_series('loans', 9)
    deposit, deposit_label = rate_series('deposits', 8)
    d = pd.read_excel(OUT / 'expectations.xlsx', sheet_name=11, header=None)
    columns = [j for j in range(1, len(d.columns)) if isinstance(d.iloc[1,j], datetime)
               and pd.Timestamp('2022-01-01') <= d.iloc[1,j] <= pd.Timestamp('2024-11-01')]
    dates = pd.DatetimeIndex([d.iloc[1,j] for j in columns])
    assert dates.equals(pd.date_range('2022-01-01','2024-11-01',freq='MS'))
    assert 'потребительских настроений' in str(d.iloc[133,0]).lower()
    assert 'откладывать, беречь' == str(d.iloc[213,0]).strip().lower()
    survey = pd.DataFrame({'confidence':pd.to_numeric(d.iloc[133,columns]).to_numpy(),
                           'save_preference':pd.to_numeric(d.iloc[213,columns]).to_numpy()},index=dates)
    assert abs(survey.loc['2024-11-01','save_preference']-53.01980198)<1e-6
    history = pd.DataFrame(index=pd.date_range('2022-01-01','2024-12-01',freq='MS'))
    history['loan_rate']=loan.reindex(history.index)
    history['deposit_rate']=deposit.reindex(history.index)
    history = history.join(survey)
    history.index.name='date'
    # The November workbook has no December survey. Every selected forecast uses a lag.
    history.to_csv(OUT / 'history.csv')
    source_urls = {'loans.xlsx':'https://www.cbr.ru/vfs/statistics/pdko/int_rat/loans_ind_new.xlsx',
                   'deposits.xlsx':'https://www.cbr.ru/vfs/statistics/pdko/int_rat/deposits.xlsx',
                   'expectations.xlsx':'https://www.cbr.ru/Collection/Collection/File/54835/Infl_exp_24-11.xlsx'}
    manifest = dict(retrieved_at=datetime.now(timezone.utc).isoformat(),
                    sources=[dict(file=name,url=url,sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest(),
                                  tls_verified=False,historical_vintage_verified=False,available_at=None)
                             for name,url in source_urls.items()],
                    definitions={'loan_rate':loan_label,'deposit_rate':deposit_label,
                                 'confidence':'Consumer sentiment index, infOM survey, row 134 of source sheet 12',
                                 'save_preference':'Prefer saving rather than expensive purchases, percent respondents, row 214'},
                    excluded='Absolute credit issuance and deposit flows not present in these selected sheets; no fabricated payment dates or demographic exposures.',
                    survey_reference_page='https://www.cbr.ru/analytics/dkp/inflationary_expectations/Infl_exp_24-11/',
                    vintage_warning='Rates are current historical snapshots. November 2024 survey workbook contains retrospective history; earlier versions not verified. Lags do not prove as-of availability.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Financial histories extracted; November savings preference independently reconciled.')


if __name__=='__main__': main()
