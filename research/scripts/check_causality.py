"""Verify that future mutations do not change either forecast for a fixed origin."""
import json
import numpy as np
from backtest import ROOT, load_panel, panel_forecast, growth_forecast

cfg = json.loads((ROOT/'configs/backtest.json').read_text(encoding='utf-8'))
_, _, cats, values = load_panel(32)
target = cats.index(cfg['target'])
origin = 17
changed = values.copy()
changed[:, origin+1:, :] *= 1000
for h in [1, 2, 3, 6]:
    before, count = panel_forecast(values, origin, h, target, cfg)
    after, _ = panel_forecast(changed, origin, h, target, cfg)
    assert before is not None and count >= cfg['min_training_origins']
    np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(growth_forecast(values, origin, h, target, 3), growth_forecast(changed, origin, h, target, 3), rtol=1e-12, atol=1e-8)
print('Passed: mutations after the forecast origin do not affect Ridge or seasonal-growth predictions.')
