"""Starter online detector; retrospective segmentations are not early warning evidence."""
import numpy as np
import pandas as pd
from backtest import ROOT, load_panel

def main():
    ids, dates, cats, values = load_panel()
    logged = np.log(values)
    # Removing the current common component is allowed only after that month is observed.
    local = logged - np.median(logged, axis=0, keepdims=True)
    innovations = np.diff(local, axis=1)
    records = []
    for t in range(7, len(dates)):
        history = innovations[:, :t-1, :]
        center = np.median(history, axis=1)
        scale = np.maximum(1.4826*np.median(np.abs(history-center[:, None, :]), axis=1), 0.01)
        score = np.max(np.abs((innovations[:, t-1, :]-center)/scale), axis=1)
        records.append(pd.DataFrame({'territory_id': ids, 'observed_month': dates[t],
            'signal_available_after_month': dates[t], 'score': score, 'alert': score > 4.0}))
    out = ROOT / 'results/change_detection'
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(records).to_parquet(out / 'online_signals.parquet', index=False)
    print('Starter signals saved. No labelled-event precision, recall or advance warning claimed.')

if __name__ == '__main__':
    main()
