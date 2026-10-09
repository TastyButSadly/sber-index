"""Paired comparisons, calibration-only ensembles and an auditable Russian report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backtest import ROOT, load_panel
from forecast_common import factor_predict

OUT=ROOT/'results/hypothesis_analysis'
KEYS=['territory_id','origin','date','horizon']
WINDOWS=['calibration_jan_may','replication_jun_sep','stress_oct_nov','december_audit']

def window(dates):
    return np.select([dates<=pd.Timestamp('2024-05-01'), dates<=pd.Timestamp('2024-09-01'), dates<=pd.Timestamp('2024-11-01')],WINDOWS[:3],default=WINDOWS[3])

def score(frame):
    rows=[]
    for win,g in frame.groupby('window'):
        y=g.actual.to_numpy();p=g.prediction.to_numpy();s=g.seasonal_value.to_numpy()
        gy=y/s-1;gp=p/s-1
        rows.append({'window':win,'n':len(g),'folds':g[['origin','horizon']].drop_duplicates().shape[0],
            'mae_rub':np.mean(np.abs(y-p)), 'r2_yoy':1-np.sum((gy-gp)**2)/np.sum((gy-gy.mean())**2)})
    return rows

def paired(a,b):
    return a.merge(b[KEYS+['prediction']],on=KEYS,suffixes=('','_other'),validate='one_to_one')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/hypotheses/predictions.parquet'
    metrics=pd.read_csv(ROOT/'results/hypotheses/metrics.csv')
    models=metrics.model.unique()
    def load(name):
        d=pd.read_parquet(source,filters=[('model','==',name)])
        d=d[d.horizon.isin([1,2,3])].copy();d['window']=window(d.date)
        return d
    base=load('ridge_seasonal_ensemble')
    ridge=load('weighted_panel_ridge')
    selection=json.loads((ROOT/'results/hypotheses/selection.json').read_text(encoding='utf8'))
    rows=[];monthly=[];comparisons=[]
    def append(name,d):
        rows.extend(dict(model=name,**m) for m in score(d))
        for (win,date,h),g in d.groupby(['window','date','horizon']):
            monthly.append({'model':name,'window':win,'date':str(date.date()),'horizon':int(h),'n':len(g),'mae_rub':np.mean(np.abs(g.actual-g.prediction))})
    for model in models:
        d=load(model)
        if len(d):append(model,d)
    growth_keys=load('growth_space_ridge')[KEYS]
    append('ensemble_on_growth_folds',base.merge(growth_keys,on=KEYS,validate='one_to_one'))
    formula_file=ROOT/'results/formula_variants/predictions.parquet'
    if formula_file.exists():
        formula=pd.read_parquet(formula_file)
        formula=formula[formula.horizon.isin([1,2,3])].copy()
        formula['window']=window(formula.date)
        for name,d in formula.groupby('model'):append(name,d)
    # Bootstrap paired differences by municipality region; holds evaluation months fixed.
    regions=pd.read_parquet(ROOT/'data/processed/municipalities_2024.parquet')[['territory_id','region_code']]
    rng=np.random.default_rng(42)
    tests=[('uni_fixed_a30_q1.5','multi_fixed_a30_q1.5'),
           ('multi_fixed_a30_q0','multi_fixed_a30_q1.5'),
           ('weighted_panel_ridge','catboost_multi'),
           ('weighted_panel_ridge','ridge_seasonal_ensemble'),
           ('ridge_on_growth_folds','growth_space_ridge')]
    for a_name,b_name in tests:
        p=paired(load(a_name),load(b_name)).merge(regions,on='territory_id',validate='many_to_one')
        p['delta']=np.abs(p.actual-p.prediction_other)-np.abs(p.actual-p.prediction)
        for win,g in p.groupby('window'):
            grouped=g.groupby('region_code').delta.agg(['sum','count'])
            draws=rng.integers(len(grouped),size=(2000,len(grouped)))
            diffs=grouped['sum'].to_numpy()[draws].sum(axis=1)/grouped['count'].to_numpy()[draws].sum(axis=1)
            comparisons.append({'reference':a_name,'candidate':b_name,'window':win,'n':len(g),
                'delta_mae_candidate_minus_reference':g.delta.mean(),'ci_low':np.quantile(diffs,.025),'ci_high':np.quantile(diffs,.975)})
    foundation_frames={}
    ensemble_selection={}
    for mode in ['multivariate','univariate','cross','residual']:
        folder=ROOT/f'results/chronos_{mode}'
        metadata=folder/'run.json'
        if not metadata.exists():continue
        meta=json.loads(metadata.read_text())
        if not meta.get('complete'):continue
        d=pd.read_parquet(folder/'predictions.parquet')
        d=d[d.horizon.isin([1,2,3])].copy();d['window']=window(d.date)
        name=f'chronos_{mode}';append(name,d);foundation_frames[name]=d
        p=paired(base,d)
        calibration=p.window=='calibration_jan_may'
        weights=[0,.05,.1,.2,.3,.5,.7,1.0]
        loss={w:np.mean(np.abs(p.loc[calibration,'actual']-((1-w)*p.loc[calibration,'prediction']+w*p.loc[calibration,'prediction_other']))) for w in weights}
        selected=min(loss,key=loss.get)
        ensemble_selection[name]={'chronos_weight':selected,'calibration_mae':loss}
        mixed=p.copy();mixed.prediction=(1-selected)*p.prediction+selected*p.prediction_other
        append(name+'_ensemble',mixed)
        for win,g in p.groupby('window'):
            log_a=np.log(g.prediction)-np.log(g.actual)
            log_b=np.log(g.prediction_other)-np.log(g.actual)
            comparisons.append({'reference':'ridge_seasonal_ensemble','candidate':name,'window':win,'n':len(g),
                'delta_mae_candidate_minus_reference':np.mean(np.abs(g.actual-g.prediction_other)-np.abs(g.actual-g.prediction)),
                'log_error_correlation':np.corrcoef(log_a,log_b)[0,1]})
    # Joint selection of foundation variant and weight uses calibration only.
    if foundation_frames:
        best=min(ensemble_selection,key=lambda n:ensemble_selection[n]['calibration_mae'][ensemble_selection[n]['chronos_weight']])
        p=paired(base,foundation_frames[best]);w=ensemble_selection[best]['chronos_weight']
        p.prediction=(1-w)*p.prediction+w*p.prediction_other
        append('chronos_selected_ensemble',p)
        p.to_parquet(OUT/'selected_ensemble_predictions.parquet',index=False)
        ensemble_selection['selected_variant']=best
    # Future common factor is used ONLY for this upper-bound diagnostic, never a valid forecast.
    ids,dates,cats,values=load_panel();target=cats.index('Все категории');factor=np.median(np.log(values),axis=0)[:,target]
    for name in ['uni_fixed_a30_q1.5','multi_fixed_a30_q1.5','weighted_panel_ridge']:
        d=load(name);offset={}
        for o,h in d[['origin','horizon']].drop_duplicates().itertuples(index=False,name=None):
            i=dates.get_loc(o);offset[o,h]=factor[i+h]-factor_predict(factor,i,h,selection['factor'])
        d.prediction=d.prediction*np.exp([offset[o,h] for o,h in zip(d.origin,d.horizon)])
        append('ORACLE_factor_diagnostic_'+name,d)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False)
    (OUT/'ensemble_selection.json').write_text(json.dumps(ensemble_selection,ensure_ascii=False,indent=2),encoding='utf8')
    pilot_rows=[]
    pilot_folder=ROOT/'results/tabpfn35_panel_pilot'
    if (pilot_folder/'run.json').exists():
        pilot=pd.read_parquet(pilot_folder/'predictions.parquet');pilot['window']=window(pilot.date)
        for name,d in pilot.groupby('model'):
            pilot_rows.extend(dict(model=name,**m) for m in score(d))
        test_keys=pilot[KEYS].drop_duplicates()
        for name in ['weighted_panel_ridge','ridge_seasonal_ensemble']:
            d=load(name).merge(test_keys,on=KEYS,validate='one_to_one')
            pilot_rows.extend(dict(model=name+'_on_pilot_keys',**m) for m in score(d))
        pd.DataFrame(pilot_rows).to_csv(OUT/'tabpfn_paired_pilot.csv',index=False)
    table=summary.pivot(index='model',columns='window',values='mae_rub').reindex(columns=WINDOWS)
    selected_models=['seasonal_growth_ema','univariate_ridge','multivariate_ridge','weighted_panel_ridge',
        'ridge_seasonal_ensemble','catboost_multi','hierarchical_3','municipality_bias','joint_ridge',
        'pls_rank2','pls_rank4','pls_rank6','growth_space_ridge','ridge_on_growth_folds','ensemble_on_growth_folds']
    selected_models += [n for n in table.index if n.startswith('chronos_')]
    lines=['# Проверка гипотез по прогнозированию СберИндекса', '',
        '> Дополнение после аудита: [repository_audit.md](repository_audit.md). Публичные «июнь–сентябрь» — origins, а здесь — целевые месяцы. Прямое сравнение 950.66 с публичными 762 некорректно; на публичных ключах наш ансамбль даёт 818.09. Модели factor_only и Chronos в двух проектах также различаются.', '',
        'Дата: 8 октября 2026. Все основные расчёты выполнены локально на CPU. Полная панель: 2 016 МО × 24 месяца × 6 категорий.', '',
        '## Протокол', '',
        'Прогноз total на горизонтах 1, 2, 3 месяца: июнь–сентябрь — 12 сочетаний origin/горизонта (24 192 ошибки); октябрь–ноябрь — 6 (12 096 ошибок). Параметры и веса ансамблей выбирались по целям до июня 2024. Калибровочных сочетаний для Ridge только шесть. Июнь–ноябрь уже обсуждались в пользовательской заметке, поэтому это проверка воспроизведения и устойчивости, а не новый blind holdout.', '',
        'Первичная калибровка выбрала общий фактор '+selection['factor']+'. Его экстраполяция не использует фактическое будущее. Масштабные веса относятся к средним расходам на жителя, а не к размеру населения или обороту МО.', '',
        '## Результаты', '',
        '| Метод | Январь–май, калибровка | Июнь–сентябрь | Октябрь–ноябрь | Декабрь |',
        '|---|---:|---:|---:|---:|']
    for name in selected_models:
        if name not in table.index:continue
        vals=['—' if pd.isna(v) else f'{v:.1f}' for v in table.loc[name]]
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['', 'Все значения — MAE в рублях. Growth-space доступен на меньшем числе ранних сочетаний (10 вместо 12 в июне–сентябре); его корректный ориентир — ridge_on_growth_folds. Остальные основные модели сравниваются на тех же данных. Декабрь выделен отдельно и тоже уже открыт анализу.', '',
        '## Выводы', '',
        'Ансамбль Ridge с сезонным EMA улучшил MAE относительно каждого из его компонентов в июне–сентябре и октябре–ноябре. Это наиболее устойчивое улучшение среди проверенных классических моделей.', '',
        'При одинаковых alpha=30 и q=1.5 шесть категорий ухудшили MAE относительно total на 91.3 рубля в июне–сентябре (региональный bootstrap: от +52.6 до +128.5), а в октябре–ноябре разница −6.6 рубля не отделяется от нуля. Веса q=1.5 против q=0 дали +23.0 рубля в первом окне и −5.0 во втором. Устойчивого преимущества категорий или масштабных весов в этой реализации нет.', '',
        'CatBoost, PLS, эксперты, персональная поправка и совместная модель горизонтов не улучшили выбранный сезонно-панельный ансамбль на обоих окнах. Growth-space следует сравнивать с ensemble_on_growth_folds: набор ранних прогнозов у него короче.', '',
        'Все четыре режима Chronos завершены. Режим лог-отклонений дал 1191.6 / 947.6 рубля в двух основных окнах, исходный multivariate — 1876.8 / 890.6. Калибровка выбрала нулевой вес каждого режима Chronos в ансамбле. Проверены уникальность ключей, конечность прогнозов и идентичность фактических значений: 82 656 прогнозов на режим, 41 сочетание origin/горизонта, 2 016 МО на каждое.', '',
        '## Что проверено', '',
        '- Совместные признаки шести категорий против признаков только total при одинаковых alpha, q и общем факторе.',
        '- q = 0, 0.5, 1, 1.5 при alpha = 30; дополнительно alpha = 3, 30, 300 с выбором по раннему окну.',
        '- Ridge против CatBoost MAE; PLS рангов 2, 4, 6; общая модель горизонтов; эксперты в 2, 3, 5 группах; shrinkage-поправка по МО.',
        '- Альтернативная постановка через YoY growth и взвешенную медиану общего роста; фиксированный ансамбль с основной моделью.',
        '- EMA сезонного роста, прогнозы общего фактора по последнему YoY, медиане трёх месяцев, EMA и затухающему тренду YoY.',
        '- Chronos-2-small: отдельно total, шесть категорий одного МО, cross-learning между соседними в порядке ID группами МО (96 рядов, то есть 16 МО на batch), локальные лог-отклонения от общего фактора. Для каждого режима сохранены все предсказания.', '',
        '## Ограничения интерпретации', '',
        'Набор реализаций заново построен по формулам заметки, прежнего кода нет. Это не точное воспроизведение неизвестных правил прогнозирования общего фактора, обучающих origins и весов. MAE 726.37 не воспроизведён. Число из другого публичного решения нельзя непосредственно сравнивать без его предсказаний и идентичного протокола.', '',
        'PLS проверяет низкоранговое пространство признаков с одной целью, но не является полной многовыходной reduced-rank regression. Персональные поправки оценены по fitted residuals с shrinkage; это конкретная проверенная реализация идеи, не восстановленная прежняя.', '',
        'formula_variants — дополнительная диагностика буквальной формулы без intercept и фиксированного ансамбля 70/30. Она добавлена после просмотра первых результатов и не является заранее выбранной моделью.', '',
        'ORACLE_factor_diagnostic в CSV использует фактический будущий общий фактор только для разложения источников ошибки. Его нельзя считать прогнозом, включать в ансамбль или публиковать как качество рабочей модели.', '',
        'Интервалы paired_comparisons.csv рассчитаны bootstrap по регионам при фиксированных месяцах. Они учитывают пространственную зависимость внутри региона, но не снимают проблему короткой и уже изученной временной истории.', '',
        'Горизонты 6 и 12 сохранены в исходных метриках. На 12 месяцев имеется только одна проверяемая точка после года контекста, обучающих пар для Ridge на ней нет. По ней нельзя заявлять устойчивость годового прогноза.', '',
        '## Детекция изменений', '',
        'CUSUM, Page–Hinkley и EWMA проверены на синтетике с известными датами: контроль, постоянный скачок, постепенный сдвиг, единичный выброс. Результаты — results/detector_validation/synthetic_metrics.csv. Это обнаружение после наблюдения изменений, не предсказание будущих реальных шоков. Высокая доля сигналов на контрольных рядах делает непроверенные пороги CUSUM и Page–Hinkley непригодными для утверждений о точном раннем предупреждении.', '',
        '## TabPFN-3.5', '',
        'Полный checkpoint 876 027 932 байта скачан, SHA-256 совпал с опубликованным в API. Выполнены два CPU-пилота: 128 обучающих примеров / 64 тестовых МО и 512 примеров / 256 случайно выбранных МО. Для каждой пары origin/горизонта обучающая подвыборка выбрана по seed 42 только из доступной истории. Проверены origins май и август 2024, горизонты 1, 2, 3; n_estimators=1. Используются те же шесть категорий, локальные log-отклонения и общий фактор, что у Ridge.', '',
        'Сравнение ограниченного Ridge и полного Ridge/ансамбля на ровно тех же тестовых ключах — tabpfn_paired_pilot.csv. Настройки Ridge перенесены из полной панели и не подбирались отдельно для малого количества обучающих строк. Этот пилот не является полным backtest TabPFN по стране; его MAE нельзя напрямую сопоставлять с основной таблицей.', '',
        'В расширенном пилоте TabPFN дал MAE 1034.7 / 1053.3 рубля против 1126.4 / 1070.9 у Ridge с той же обучающей подвыборкой. Полный Ridge на тех же тестовых ключах дал 1028.7 / 978.8, а сезонно-панельный ансамбль — 834.6 / 903.6. TabPFN помог относительно ограниченного Ridge, но преимущества перед текущим полным ансамблем этот пилот не показал.', '',
        '## Что осталось непроверенным', '',
        'Реальные новости и причинное предсказание будущих шоков требуют событий с датами публикации и знания фактической задержки публикации расходов. Синтетическая проверка не заменяет такую разметку.', '',
        '## Артефакты', '',
        '- configs/hypotheses.json: заранее заданная сетка и параметры альтернатив.',
        '- results/hypotheses/selection.json: решения калибровки и время работы.',
        '- results/hypotheses/predictions.parquet и results/chronos_*/predictions.parquet: предсказания.',
        '- results/hypothesis_analysis/summary.csv, monthly_metrics.csv, paired_comparisons.csv: сводки и парные эффекты.',
        '- results/hypothesis_analysis/ensemble_selection.json: веса Chronos, выбранные только по раннему окну.',
        '- scripts/hypothesis_suite.py, run_chronos.py, analyze_hypotheses.py: воспроизводимые расчёты.', '']
    (ROOT/'reports/hypotheses_report.md').write_text('\n'.join(lines),encoding='utf8')
    print(table.round(2).to_string())

if __name__=='__main__':main()
