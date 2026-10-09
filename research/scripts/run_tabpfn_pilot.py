"""CPU pilot of the full TabPFN-3.5 checkpoint, clearly distinct from the full backtest."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['ONEDNN_PRIMITIVE_CACHE_CAPACITY']='0'
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNRegressor
from backtest import ROOT,load_panel,summarize
from forecast_common import factor_predict
from panel_features import dataset

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-samples',type=int,default=128)
    parser.add_argument('--test-territories',type=int,default=64)
    parser.add_argument('--origins',nargs='+',default=['2024-05-01','2024-08-01'])
    parser.add_argument('--horizons',nargs='+',type=int,default=[1,2,3])
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output',default='results/tabpfn35_pilot')
    args=parser.parse_args()
    torch.set_num_threads(args.threads);torch.backends.mkldnn.enabled=False
    ids,dates,cats,values=load_panel();target=cats.index('Все категории')
    rng=np.random.default_rng(42)
    test_indices=np.sort(rng.choice(len(ids),args.test_territories,replace=False))
    factor=np.median(np.log(values),axis=0);residual=np.log(values)-factor[None,:,:]
    selection=json.loads((ROOT/'results/hypotheses/selection.json').read_text())
    chosen=selection['selected_ridge']['weighted_panel_ridge'];alpha=float(chosen.split('_a')[1])
    q=float(chosen.split('_q')[1].split('_')[0])
    folder=ROOT/args.output;folder.mkdir(parents=True,exist_ok=True)
    model_path=ROOT/'models/tabpfn-3.5/tabpfn-v3.5-20260909.safetensors'
    model=TabPFNRegressor(model_path=str(model_path),device='cpu',n_estimators=1,
        fit_mode='low_memory',random_state=42,inference_precision=torch.float32,
        memory_saving_mode=True,n_preprocessing_jobs=1,show_progress_bar=False)
    records=[];started=time.time()
    for origin_date in args.origins:
        o=dates.get_loc(pd.Timestamp(origin_date))
        for h in args.horizons:
            x,y,scale,xf,starts=dataset(values,residual,o,h,target,np.arange(len(cats)))
            chosen_rows=np.sort(rng.choice(len(x),min(args.train_samples,len(x)),replace=False))
            observed=x[chosen_rows];labels=y[chosen_rows]
            with threadpool_limits(args.threads):
                model.fit(pd.DataFrame(observed),labels)
                forecast=model.predict(pd.DataFrame(xf[test_indices]))
                matched=Ridge(alpha=alpha).fit(observed,labels,sample_weight=scale[chosen_rows]**q).predict(xf[test_indices])
            common=factor_predict(factor[:,target],o,h,selection['factor'])
            for name,local in [('tabpfn35_pilot',forecast),('ridge_same_training_sample',matched)]:
                records.append(pd.DataFrame({'territory_id':ids[test_indices],'origin':dates[o],
                    'date':dates[o+h],'horizon':h,'model':name,'prediction':np.exp(local+common),
                    'actual':values[test_indices,o+h,target],'seasonal_value':values[test_indices,o+h-12,target],
                    'split':'development' if dates[o+h]<=pd.Timestamp('2024-09-01') else 'late_stress'}))
            pd.concat(records).to_parquet(folder/'predictions.parquet',index=False)
            print(f'TabPFN pilot {origin_date} h={h}: {time.time()-started:.1f}s',flush=True)
    frame=pd.concat(records,ignore_index=True)
    result=summarize(frame);result.to_csv(folder/'metrics.csv',index=False)
    metadata=dict(vars(args),seed=42,test_ids=ids[test_indices].tolist(),n_estimators=1,
        model='Full TabPFN-3.5, 876027932-byte checkpoint',runtime_seconds=time.time()-started,
        warning='Limited training examples and randomly selected test municipalities. Not comparable to full-country MAE.')
    (folder/'run.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':main()
