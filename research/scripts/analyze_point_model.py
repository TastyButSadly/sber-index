"""Write paired point-model evidence, including temporal variation and limitations."""
import json
import numpy as np
import pandas as pd
from backtest import ROOT


def main():
    out=ROOT/'results/point_ensemble'
    d=pd.read_parquet(out/'predictions.parquet')
    names=['reference_ens3','bolt_ridge_seasonal_mean1','bolt_ridge_seasonal_ema']
    d=d[d.model.isin(names)]
    keys=['territory_id','origin','date','horizon']
    base=d[d.model==names[0]].set_index(keys)
    rows=[];by_h=[];by_month=[];cis=[]
    rng=np.random.default_rng(20261008)
    for name in names:
        g=d[d.model==name].set_index(keys).reindex(base.index)
        assert np.array_equal(g.actual,base.actual)
        e=np.abs(g.prediction-g.actual)
        b=np.abs(base.prediction-base.actual)
        temp=g.reset_index();temp['error']=e.to_numpy();temp['base_error']=b.to_numpy()
        public=temp[temp.origin.between('2024-06-01','2024-09-01')]
        for h,t in public.groupby('horizon'):
            by_h.append(dict(model=name,horizon=h,n=len(t),mae_rub=float(t.error.mean())))
        for month,t in public.groupby('date'):
            by_month.append(dict(model=name,target_month=str(month.date()),n=len(t),mae_rub=float(t.error.mean())))
        for window,t in [('public_origins_jun_sep',public),
                         ('targets_oct_nov',temp[temp.date.between('2024-10-01','2024-11-01')]),
                         ('later_origins',temp[temp.origin>pd.Timestamp('2024-09-01')])]:
            rows.append(dict(model=name,window=window,n=len(t),mae_rub=float(t.error.mean()),
                relative_reduction_percent=float(100*(1-t.error.mean()/t.base_error.mean()))))
        grouped=public.assign(difference=public.error-public.base_error).groupby('date').agg(
            difference_sum=('difference','sum'),n=('difference','size'))
        draw=rng.integers(0,len(grouped),size=(2000,len(grouped)))
        diff=grouped.difference_sum.to_numpy()[draw].sum(axis=1)/grouped.n.to_numpy()[draw].sum(axis=1)
        lo,hi=np.quantile(diff,[.025,.975])
        cis.append(dict(model=name,target_month_clusters=len(grouped),
            delta_mae_rub=float(public.error.mean()-public.base_error.mean()),
            bootstrap_low=float(lo),bootstrap_high=float(hi)))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'selected_metrics.csv',index=False)
    pd.DataFrame(by_h).to_csv(out/'horizon_metrics.csv',index=False)
    pd.DataFrame(by_month).to_csv(out/'target_month_metrics.csv',index=False)
    pd.DataFrame(cis).to_csv(out/'month_bootstrap.csv',index=False)
    table=metrics.pivot(index='model',columns='window',values='mae_rub')
    text='''# Новая точечная модель: Bolt + Ridge + сезонный прогноз

Собран запускаемый ансамбль, который меняет сам точечный прогноз. Предыдущий байесовский эксперимент менял только интервалы вокруг ens3 и не снижал его MAE.

## Архитектура и признаки

Для каждой категории общий фактор — медиана логарифма расходов по муниципалитетам. Локальное состояние — отклонение от этого фактора. Компоненты прогнозируют местную часть, после чего возвращается прогноз общего фактора; три рублёвых прогноза усредняются с равными весами.

1. Chronos Bolt-base: замороженная локальная модель по истории остатка категории «Все категории», медианный прогноз.
2. Панельный Ridge: отдельная обученная регрессия для каждого горизонта, alpha=3, веса обучающих примеров пропорциональны сезонному уровню расходов в степени 1.5, нормированы до среднего 1. Признаки из всех шести категорий: текущий и предыдущий остатки, сезонный остаток нужного месяца и изменение за последний месяц. Всего 24 признака; обучающие цели доступны к origin.
3. Сезонная модель: индивидуальный средний логарифмический YoY-рост за последние три наблюдения. Она также переведена в общее пространство локального остатка и фактора.

Версия **mean1** использует последний наблюдённый YoY-рост общего фактора. Версия **EMA** усредняет его прошлые YoY-изменения с весами 0.5^возраст. Это прогноз фактора из прошлого, а не фактический будущий фактор. ETS в итоговые три компоненты не вошёл.

## Проверенные результаты

Все строки сравниваются на одинаковых ключах территории, origin, целевого месяца и горизонта. Метрика — MAE расходов в рублях; 2 016 МО, категория «Все категории», h=1,2,3.

| Модель | Origins июнь–сентябрь | Цели октябрь–ноябрь | Origins октябрь–ноябрь |
|---|---:|---:|---:|
'''
    labels={names[0]:'Исходный ens3',names[1]:'Новый ансамбль mean1',names[2]:'Новый ансамбль EMA'}
    for n in names:
        text+=f'| {labels[n]} | {table.loc[n,"public_origins_jun_sep"]:.2f} | {table.loc[n,"targets_oct_nov"]:.2f} | {table.loc[n,"later_origins"]:.2f} |\n'
    text+='''
Первое окно содержит 24 192 прогноза, второе — 12 096, последнее — 6 048 (в ноябре h=2,3 уже выходят за конец наблюдений, в октябре — h=3). Окна перекрываются, их нельзя считать независимыми подтверждениями.

**mean1 выбран по ошибкам с целевыми месяцами до мая 2024 включительно** среди 16 фиксированных кандидатов, без использования июня–декабря в функции выбора. Его выигрыш в основном окне составляет около 0.83%. EMA дал снижение примерно 5.57%, но этот вариант признан лучшим после просмотра основного окна: его число — исследовательский результат, а не независимая оценка будущего качества. Вся эта история уже изучалась раньше; даже ранний выбор не превращает её в новый слепой тест.

Проверены также причинные селекторы по последним четырём уже наблюдённым целевым месяцам: общий выбор со сжатием к ens3, выбор по горизонту и среднее трёх лучших моделей. Их MAE в основном окне 757.54, 755.32 и 761.04 соответственно. Они не достигли результата EMA; пока оставлены диагностикой, равные веса не обучались на тестовом окне.

Разрезы по горизонтам и целевым месяцам сохранены отдельно. Bootstrap по целевым месяцам находится в month_bootstrap.csv; всего шесть временных кластеров, интервалы описательные и не учитывают выбор лучшего варианта из перебора.

## Реализация и проверка

`scripts/forecast_point.py` обучает Ridge и вычисляет свежий прогноз из входной истории. Состояние обученной модели сохраняется в trained_model.joblib, загружается через PointEnsemble.load; веса Bolt лежат локально и зафиксированы ревизией. Это вычисление модели, а не извлечение уже сохранённых тестовых ответов.

Повторный расчёт origin сентябрь 2024 воспроизвёл 12 096 прогнозов двух вариантов с максимальным расхождением 0.0037 ₽. Умножение всех значений после origin на 1000 не меняет локальные входы, фактор или прогноз Ridge. Изменение ещё неизвестных целевых значений не меняет набор ошибок для причинного выбора. Сохранение и загрузка обученного состояния проверены. Результаты: results/point_ensemble/checks.json.

```powershell
.\\.venv\\Scripts\\python.exe scripts/develop_point_ensemble.py
.\\.venv\\Scripts\\python.exe scripts/forecast_point.py --horizons 1 2 3 6 12
.\\.venv\\Scripts\\python.exe scripts/forecast_point.py --origin 2024-09-01 --output results/point_forecast_verification
.\\.venv\\Scripts\\python.exe scripts/check_point_model.py
.\\.venv\\Scripts\\python.exe scripts/analyze_point_model.py
```

Уже сохранён прогноз от декабря 2024: январь, февраль, март, июнь и декабрь 2025 по двум вариантам, 20 160 строк, с отдельными прогнозами компонентов. Значения 2025 года здесь не наблюдались; это демонстрация применения модели, а не оценка качества на 2025 год. Горизонты 6 и 12 исполняются, но указанные MAE их не подтверждают.

## Что ещё не включено

Календарные, макроэкономические и новостные признаки не включены в этот ансамбль. Байесовская локальная регрессия и BART как точечные компоненты не обучались в этом раунде; байесовский слой интервалов был отдельным экспериментом. Интервалы, откалиброванные вокруг прежнего ens3, нельзя автоматически переносить на новый прогноз — их нужно пересчитать.

Дата origin означает последний месяц наблюдений, даты фактической публикации неизвестны. Полная когорта 2 016 МО выбрана ретроспективно по всей истории; это ограничение исходного аудита сохраняется. Сравнение касается только total, а не пяти остальных отдельных целей. Архитектура и новые признаки требуют отдельной проверки на более длинной или ещё не просмотренной истории.

## Происхождение

Опорные прогнозы для сравнения воспроизведены из MIT-репозитория rav11l/sberindex-shocks, ревизия 956bbf10a67d9b8f2815cd62cde4239c8fa75b83; подробности в reports/repository_audit.md. Новый обучаемый Ridge и сборка local × factor — код текущего проекта. Веса Chronos Bolt-base: ревизия 5d9f166d69f47aef3401367a7b842e78fe97b121. Исходные данные и условия атрибуции СберИндекса указаны в README.
'''
    (ROOT/'reports/point_ensemble_report.md').write_text(text,encoding='utf8')
    print(pd.DataFrame(cis).to_string(index=False))


if __name__=='__main__':main()
