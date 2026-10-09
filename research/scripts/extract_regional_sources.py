"""Read official spreadsheets into JSON using the bundled analysis runtime."""
import json
import re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'data/external/regional';out.mkdir(parents=True,exist_ok=True)
    price=ROOT/'data/external/regional_prices.xlsx';wage=ROOT/'data/external/regional_wages.xlsx'
    rows=[]
    workbook=pd.ExcelFile(price)
    for sheet in workbook.sheet_names:
        match=re.fullmatch(r'(\d{2})\((\d{4})\)',sheet)
        if not match or int(match[2])>2024:continue
        date=f'{match[2]}-{match[1]}-01'
        table=pd.read_excel(workbook,sheet_name=sheet,header=None)
        for _,r in table.iloc[4:].iterrows():
            if not isinstance(r.iloc[0],(int,float)) or pd.isna(r.iloc[0]):continue
            if not all(pd.notna(v) and isinstance(v,(int,float)) for v in r.iloc[2:6]):continue
            rows.append(dict(date=date,territory_code=int(r.iloc[0]),source_region_name=str(r.iloc[1]).strip(),
                cpi_all=float(r.iloc[2]),cpi_food=float(r.iloc[3]),cpi_nonfood=float(r.iloc[4]),cpi_services=float(r.iloc[5])))
    (out/'prices_raw.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf8')
    table=pd.read_excel(wage,sheet_name='с 2019',header=None)
    months=['январь','февраль','март','апрель','май','июнь','июль','август','сентябрь','октябрь','ноябрь','декабрь']
    columns=[];year=None
    for j in range(1,table.shape[1]):
        match=re.search(r'20\d{2}',str(table.iloc[1,j]))
        if match:year=int(match[0])
        month=str(table.iloc[2,j]).strip().lower()
        if year is not None and year<=2024 and month in months:columns.append((j,f'{year}-{months.index(month)+1:02d}-01'))
    rows=[]
    for _,r in table.iloc[3:].iterrows():
        if pd.isna(r.iloc[0]):continue
        for j,date in columns:
            if isinstance(r.iloc[j],(int,float)) and pd.notna(r.iloc[j]):
                rows.append(dict(date=date,source_region_name=str(r.iloc[0]).strip(),wage_rub=float(r.iloc[j])))
    (out/'wages_raw.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf8')
    print('Read regional prices and wages; observations after 2024 excluded',flush=True)


if __name__=='__main__':main()
