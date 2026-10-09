"""Prespecified ablations of the hypotheses in the user-provided note.

Hyperparameters and shared factor forecasts are chosen only on target months
through May 2024. June-November were inspected in the previous note and are
replication/stress windows, not a fresh blind holdout.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits
from catboost import CatBoostRegressor
from backtest import ROOT, load_panel
from forecast_common import factor_predict
from panel_features import dataset

OUT = ROOT / 'results/hypotheses'

def split_name(date, cfg):
    if date <= pd.Timestamp(cfg['calibration_end']): return 'calibration_jan_may'
    if date <= pd.Timestamp(cfg['reproduction_end']): return 'replication_jun_sep'
    if date <= pd.Timestamp(cfg['stress_end']): return 'stress_oct_nov'
    return 'december_audit'

def weighted_median(x, w):
    order = np.argsort(x)
    return x[order[np.searchsorted(np.cumsum(w[order]), 0.5*w.sum())]]

def metrics(pred, values, target, dates, cfg):
    rows = []
    for model, cube in pred.items():
        for window in ['calibration_jan_may','replication_jun_sep','stress_oct_nov','december_audit']:
            for horizon in ['1','2','3','6','12','1_2_3']:
                hs = [1,2,3] if horizon == '1_2_3' else [int(horizon)]
                actuals, forecasts, seasons, folds = [], [], [], 0
                for (origin, h), forecast in cube.items():
                    if h not in hs or split_name(dates[origin+h], cfg) != window: continue
                    actuals.append(values[:, origin+h, target]); forecasts.append(forecast)
                    seasons.append(values[:, origin+h-12, target]); folds += 1
                if not actuals: continue
                actual, forecast, seasonal = map(np.concatenate, (actuals, forecasts, seasons))
                g, pg = actual/seasonal-1, forecast/seasonal-1
                rows.append({'model':model,'window':window,'horizon':horizon,'folds':folds,'n':len(actual),
                    'mae_rub':float(np.mean(np.abs(actual-forecast))),
                    'r2_level':float(1-np.sum((actual-forecast)**2)/np.sum((actual-actual.mean())**2)),
                    'r2_yoy':float(1-np.sum((g-pg)**2)/np.sum((g-g.mean())**2))})
    return pd.DataFrame(rows)

def calibration_mae(cube, values, target, dates, cfg):
    errors = [np.abs(values[:, o+h, target]-p) for (o,h), p in cube.items()
              if h <= 3 and dates[o+h] <= pd.Timestamp(cfg['calibration_end'])]
    return float(np.mean(np.concatenate(errors))) if errors else float('inf')

def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT/'configs/hypotheses.json').read_text(encoding='utf-8'))
    (OUT/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    ids, dates, cats, values = load_panel()
    target = cats.index('Все категории')
    logs = np.log(values)
    factor = np.median(logs, axis=0)
    residual = logs-factor[None,:,:]
    # Baselines and calibration of the future common factor.
    factor_predictions = {name:{} for name in cfg['factor_methods']}
    pred = {'seasonal_naive':{},'seasonal_growth_median3':{},'seasonal_growth_ema':{},'factor_only':{}}
    from backtest import growth_forecast
    all_folds = [(o,h) for o in range(11,len(dates)-1) for h in cfg['horizons'] if o+h<len(dates)]
    for o,h in all_folds:
        seasonal = values[:,o+h-12,target]
        pred['seasonal_naive'][o,h] = seasonal
        pred['seasonal_growth_median3'][o,h] = growth_forecast(values,o,h,target,3)
        if o >= 12:
            changes = logs[:,12:o+1,target]-logs[:,:o-11,target]
            w = 0.5**np.arange(changes.shape[1]-1,-1,-1)
            pred['seasonal_growth_ema'][o,h] = seasonal*np.exp(np.average(changes,weights=w,axis=1))
        else: pred['seasonal_growth_ema'][o,h] = seasonal
        for name in cfg['factor_methods']:
            f = factor_predict(factor[:,target],o,h,name)
            factor_predictions[name][o,h] = np.exp(f+residual[:,o+h-12,target])
    factor_scores = {name:calibration_mae(cube,values,target,dates,cfg) for name,cube in factor_predictions.items()}
    chosen_factor = min(factor_scores,key=factor_scores.get)
    pred['factor_only'] = factor_predictions[chosen_factor]
    factor_path = {key:factor_predict(factor[:,target],*key,chosen_factor) for key in all_folds}
    print('Factor chosen on Jan-May only:',chosen_factor,factor_scores,flush=True)
    grid = {}
    for columns, name in [(np.array([target]),'uni'),(np.arange(len(cats)),'multi')]:
        for q in cfg['q_grid']:
            for alpha in cfg['alpha_grid']:
                cube = {}
                for o,h in all_folds:
                    ds = dataset(values,residual,o,h,target,columns)
                    if ds is None: continue
                    x,y,scale,xf,starts = ds
                    w = scale**q;w/=w.mean()
                    fitted = Ridge(alpha=alpha).fit(x,y,sample_weight=w)
                    cube[o,h] = np.exp(fitted.predict(xf)+factor_path[o,h])
                grid[f'{name}_q{q:g}_a{alpha:g}'] = cube
        print('Finished Ridge grid',name,round(time.time()-started,1),'sec',flush=True)
    tuning = pd.DataFrame([{'config':name,'calibration_mae':calibration_mae(cube,values,target,dates,cfg)} for name,cube in grid.items()])
    tuning.to_csv(OUT/'ridge_tuning.csv',index=False)
    chosen = {}
    for name,prefix in [('univariate_ridge','uni_'),('multivariate_ridge','multi_q0_'),('weighted_panel_ridge','multi_')]:
        options = tuning[tuning.config.str.startswith(prefix)]
        if name == 'weighted_panel_ridge': options = options[~options.config.str.startswith('multi_q0_')]
        best = options.sort_values('calibration_mae').iloc[0]['config']
        chosen[name]=best;pred[name]=grid[best]
    # Fixed ablations: identical alpha and factor, varying q and category count.
    for q in cfg['q_grid']:
        pred[f'multi_fixed_a30_q{q:g}'] = grid[f'multi_q{q:g}_a30']
    pred['uni_fixed_a30_q1.5'] = grid['uni_q1.5_a30']
    base_config = chosen['weighted_panel_ridge']
    q = float(base_config.split('_q')[1].split('_')[0]); alpha=float(base_config.split('_a')[1])
    baseline = pred['weighted_panel_ridge']
    for name in ['catboost_multi','municipality_bias','joint_ridge','growth_space_ridge']:
        pred[name]={}
    for k in cfg['expert_groups']:pred[f'hierarchical_{k}']={}
    for rank in cfg['pls_ranks']:pred[f'pls_rank{rank}']={}
    for fold,(o,h) in enumerate(all_folds):
        ds = dataset(values,residual,o,h,target,np.arange(len(cats)))
        if ds is None:continue
        x,y,scale,xf,starts=ds;w=scale**q;w/=w.mean()
        model=Ridge(alpha=alpha).fit(x,y,sample_weight=w)
        local=model.predict(xf)
        params=cfg['catboost'].copy()
        booster=CatBoostRegressor(**params,random_seed=cfg['seed'],verbose=False,thread_count=2,allow_writing_files=False)
        booster.fit(x,y,sample_weight=w)
        pred['catboost_multi'][o,h]=np.exp(booster.predict(xf)+factor_path[o,h])
        # Shrinkage bias from historical fitted residuals; predictive verification is via future folds.
        errors=(y-model.predict(x)).reshape(len(starts),len(ids))
        bias=errors.mean(axis=0)*len(starts)/(len(starts)+cfg['municipality_bias_prior_months'])
        pred['municipality_bias'][o,h]=np.exp(local+bias+factor_path[o,h])
        for k in cfg['expert_groups']:
            train_size=np.concatenate([values[:,s,target] for s in starts])
            cut=np.quantile(train_size,np.arange(1,k)/k)
            train_groups=np.digitize(train_size,cut);test_groups=np.digitize(values[:,o,target],cut)
            expert=local.copy()
            for group in range(k):
                mask=train_groups==group;future=test_groups==group
                specialist=Ridge(alpha=alpha).fit(x[mask],y[mask],sample_weight=w[mask])
                expert[future]=specialist.predict(xf[future])
            blended=(1-cfg['expert_weight'])*local+cfg['expert_weight']*expert
            pred[f'hierarchical_{k}'][o,h]=np.exp(blended+factor_path[o,h])
        for rank in cfg['pls_ranks']:
            pls=PLSRegression(n_components=rank).fit(x,y)
            pred[f'pls_rank{rank}'][o,h]=np.exp(pls.predict(xf).reshape(-1)+factor_path[o,h])
        if h<=3:
            joint_x=[];joint_y=[];joint_w=[]
            for hh in [1,2,3]:
                other=dataset(values,residual,o,hh,target,np.arange(len(cats)))
                if other is None:continue
                xx,yy,ss,_,_=other
                indicators=np.tile(np.eye(3)[hh-1],(len(xx),1))
                joint_x.append(np.c_[xx,indicators,xx*(hh-2)])
                joint_y.append(yy);joint_w.append(ss**q)
            joint_weights=np.concatenate(joint_w);joint_weights/=joint_weights.mean()
            joint=Ridge(alpha=alpha).fit(np.concatenate(joint_x),np.concatenate(joint_y),sample_weight=joint_weights)
            test=np.c_[xf,np.tile(np.eye(3)[h-1],(len(xf),1)),xf*(h-2)]
            pred['joint_ridge'][o,h]=np.exp(joint.predict(test)+factor_path[o,h])
        # Alternative YoY-growth space; needs 12 lagged months for all observed categories.
        growth=logs[:,12:o+1]-logs[:,:o-11]
        if growth.shape[1]>=h+2:
            common=np.array([[weighted_median(growth[:,t,c],values[:,t,c]) for c in range(len(cats))] for t in range(growth.shape[1])])
            gr=growth-common[None,:,:]
            xx=[];yy=[];ww=[]
            for s in range(1,gr.shape[1]-h):
                xx.append(np.c_[gr[:,s],gr[:,s-1],gr[:,s]-gr[:,s-1]])
                yy.append(gr[:,s+h,target]);ww.append(values[:,s+h,target]**q)
            gx=np.concatenate(xx);gy=np.concatenate(yy);gw=np.concatenate(ww);gw/=gw.mean()
            gm=Ridge(alpha=alpha).fit(gx,gy,sample_weight=gw)
            gf=np.c_[gr[:,-1],gr[:,-2],gr[:,-1]-gr[:,-2]]
            future_common=np.median(common[-3:,target])
            pred['growth_space_ridge'][o,h]=values[:,o+h-12,target]*np.exp(gm.predict(gf)+future_common)
        if fold%5==0:print('Alternative models fold',fold+1,'/',len(all_folds),round(time.time()-started,1),'sec',flush=True)
    # Choose local-model/seasonal blend on calibration only.
    seasonal=pred['seasonal_growth_ema']
    blends={}
    for weight in cfg['ensemble_weights']:
        blends[weight]={key:weight*p+(1-weight)*seasonal[key] for key,p in baseline.items()}
    blend_scores={weight:calibration_mae(cube,values,target,dates,cfg) for weight,cube in blends.items()}
    blend_weight=min(blend_scores,key=blend_scores.get)
    pred['ridge_seasonal_ensemble']=blends[blend_weight]
    pred['ridge_seasonal_fixed_0.7']=blends[0.7]
    # Verify factor sensitivity with the same trained local model, avoiding additional tuning.
    for method in cfg['factor_methods']:
        pred[f'weighted_factor_{method}']={key:p*np.exp(factor_predict(factor[:,target],*key,method)-factor_path[key]) for key,p in baseline.items()}
    # Paired intersection within each family; growth formulation gets its own matched baseline.
    for key,p in pred['growth_space_ridge'].items():
        pred.setdefault('ridge_on_growth_folds',{})[key]=baseline[key]
        pred.setdefault('growth_ridge_fixed_0.3',{})[key]=0.3*p+0.7*baseline[key]
    metrics(pred,values,target,dates,cfg).to_csv(OUT/'metrics.csv',index=False)
    # Preserve predictions for independent ensemble analysis (float64, all municipality IDs).
    rows=[]
    for name,cube in pred.items():
        for (o,h),p in cube.items():
            rows.append(pd.DataFrame({'territory_id':ids,'origin':dates[o],'date':dates[o+h],'horizon':h,
                'model':name,'prediction':p,'actual':values[:,o+h,target],
                'seasonal_value':values[:,o+h-12,target],'split':split_name(dates[o+h],cfg)}))
    pd.concat(rows,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    grid_metrics=metrics(grid,values,target,dates,cfg)
    grid_metrics.to_csv(OUT/'ridge_grid_metrics.csv',index=False)
    selection={'factor':chosen_factor,'factor_calibration_mae':factor_scores,'selected_ridge':chosen,
        'seasonal_ensemble_panel_weight':blend_weight,'ensemble_calibration_mae':blend_scores,
        'runtime_seconds':time.time()-started,'n_territories':len(ids),
        'selection_window':'Target months through 2024-05 only; horizons 1,2,3',
        'warning':'June-November already inspected in user note; no fresh blind holdout'}
    (OUT/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)
    print(metrics(pred,values,target,dates,cfg).query("horizon == '1_2_3'").to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
