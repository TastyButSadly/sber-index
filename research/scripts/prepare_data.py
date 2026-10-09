"""Validate the original contest snapshot and build an ML panel without imputation."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def main():
    source = ROOT / 'data/raw/hackathon/hackathonlicence/consumption.parquet'
    df = pd.read_parquet(source)
    df = df.rename(columns={'value': 'consumption'})
    required = ['territory_id', 'date', 'category', 'consumption']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Unexpected schema: {list(df.columns)}')
    df = df[required].copy()
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
    keys = ['territory_id', 'date', 'category']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate territory/date/category keys')
    panel = df.pivot(index=['territory_id', 'date'], columns='category', values='consumption').sort_index()
    panel.columns.name = None
    months = sorted(df['date'].unique())
    territories = df['territory_id'].nunique()
    panel.reset_index().to_parquet(ROOT / 'data/processed/panel_unbalanced.parquet', index=False)
    counts = df.groupby('territory_id').size()
    expected_per_id = len(months) * df['category'].nunique()
    complete_ids = counts[counts == expected_per_id].index
    balanced = panel.loc[panel.index.get_level_values('territory_id').isin(complete_ids)]
    balanced.reset_index().to_parquet(ROOT / 'data/processed/panel.parquet', index=False)
    counts[counts < expected_per_id].rename('observed_cells').to_csv(ROOT / 'reports/excluded_territories.csv')
    dictionary = pd.read_excel(ROOT / 'data/raw/t_dict_municipal_districts.xlsx')
    current = dictionary[(dictionary.year_from <= 2024) & (dictionary.year_to > 2024)]
    if current.territory_id.duplicated().any():
        raise ValueError('2024 municipality dictionary has duplicate territory_id')
    current.to_parquet(ROOT / 'data/processed/municipalities_2024.parquet', index=False)
    audit = {
        'source': str(source.relative_to(ROOT)), 'rows': len(df), 'territories': territories,
        'months': len(months), 'start': str(pd.Timestamp(months[0]).date()),
        'end': str(pd.Timestamp(months[-1]).date()), 'categories': sorted(df['category'].unique()),
        'missing_values': int(df['consumption'].isna().sum()),
        'nonpositive_values': int((df['consumption'] <= 0).sum()),
        'expected_rows': territories * len(months) * df['category'].nunique(),
        'panel_missing_cells': int(panel.isna().sum().sum()),
        'missing_calendar_category_cells': territories * len(months) * df['category'].nunique() - len(df),
        'balanced_territories': len(complete_ids), 'balanced_rows': len(balanced),
        'excluded_territories': territories - len(complete_ids),
        'unmatched_balanced_ids_in_2024_dictionary': sorted(set(complete_ids)-set(current.territory_id)),
        'cohort_warning': 'Complete-series cohort selected retrospectively. Compare causal expanding cohorts separately before claiming deployment generalization.',
        'units': 'Average cashless consumption of residents, RUB; not aggregate municipal turnover',
    }
    if len(months) != len(pd.date_range(months[0], months[-1], freq='MS')):
        raise ValueError('Missing calendar months')
    for file in (ROOT / 'data/raw').rglob('*.parquet'):
        table = pd.read_parquet(file)
        audit.setdefault('files', {})[str(file.relative_to(ROOT))] = {
            'rows': len(table), 'columns': list(table.columns),
        }
    (ROOT / 'reports/data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
