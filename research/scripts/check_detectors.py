"""Synthetic validation of causal detectors, with known event times and controls."""
import json
import numpy as np
import pandas as pd
from backtest import ROOT

def detect(series, method):
    # Fixed calibration uses only the first 12 observations of each sequence.
    mean = np.mean(series[:12]); scale = max(np.std(series[:12],ddof=1),0.01)
    z = (series-mean)/scale
    upper=lower=0.0; fast=slow=0.0; running=0.0; ph=low=high=0.0
    alerts=[]
    for t in range(12,len(series)):
        v=z[t]
        if method=='cusum':
            upper=max(0,upper+v-0.5);lower=max(0,lower-v-0.5)
            alarm=max(upper,lower)>6
            if alarm:upper=lower=0
        elif method=='page_hinkley':
            running += (v-running)/(t-11)
            ph += v-running
            low=min(low,ph);high=max(high,ph)
            alarm=max(ph-low,high-ph)>8
            if alarm:ph=low=high=0
        elif method=='ewma':
            fast=0.4*v+0.6*fast;slow=0.05*v+0.95*slow
            alarm=abs(fast-slow)>2.5
        else:raise ValueError(method)
        if alarm:alerts.append(t)
    return alerts

def main():
    rng=np.random.default_rng(43)
    rows=[];length=48;tau=24;n=500
    for method in ['cusum','page_hinkley','ewma']:
        rng=np.random.default_rng(43)  # Same synthetic sequences for every method.
        for scenario in ['control','persistent_step','gradual_change','single_outlier']:
            detections=[];delays=[];pre=[];false=[]
            for i in range(n):
                noise=rng.normal(0,0.04,length)
                for t in range(1,length):noise[t]+=0.3*noise[t-1]
                direction=-1 if i%2 else 1
                if scenario=='persistent_step':noise[tau:]+=direction*0.2
                elif scenario=='gradual_change':noise[tau:]+=direction*0.015*np.arange(1,length-tau+1)
                elif scenario=='single_outlier':noise[tau]+=direction*0.3
                alarms=detect(noise,method)
                pre.append(any(t<tau for t in alarms))
                false.append(bool(alarms))
                hits=[t for t in alarms if tau<=t<tau+6]
                detections.append(bool(hits))
                if hits:delays.append(hits[0]-tau)
            rows.append({'method':method,'scenario':scenario,'sequences':n,
                'pre_event_alarm_fraction':np.mean(pre),
                'alarm_fraction_entire_sequence':np.mean(false),
                'detected_within_6_months':np.mean(detections) if scenario!='control' else None,
                'mean_delay_detected_months':np.mean(delays) if delays and scenario!='control' else None})
    out=ROOT/'results/detector_validation';out.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows);result.to_csv(out/'synthetic_metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'seed':43,'length_months':length,'event_month_index':tau,
        'calibration_months':12,'sequences_per_scenario':n,'noise_sd':0.04,'ar1':0.3,
        'persistent_log_shift':0.2,'gradual_log_shift_per_month':0.015,'single_outlier_log_shift':0.3,
        'warning':'Synthetic detection after observed changes; not evidence of forecasting real shocks. Thresholds not tuned.'},indent=2),encoding='utf8')
    print(result.to_string(index=False))

if __name__=='__main__':main()
