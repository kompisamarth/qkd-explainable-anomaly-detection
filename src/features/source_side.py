"""Causal source telemetry features. Explicit allowlist excludes latent truth."""
import numpy as np
import pandas as pd
from src.simulator.source_side import MINIMAL, load_source_config


def extract_source_features(observations, profile='minimal', temporal=True, environment=True, config=None):
    c = config or load_source_config()
    keys = MINIMAL if profile == 'minimal' else list(c['sensors'])
    if not environment: keys = [k for k in keys if k != 'temperature']
    times = observations.time_s.to_numpy()
    if len(times)>1 and (np.diff(times)<=0).any():
        raise ValueError('timestamps must be strictly increasing')
    dt = c['dt_s']
    if not np.isfinite(times).all() or (len(times)>1 and not np.allclose(np.diff(times),dt,rtol=0,atol=1e-8)):
        raise ValueError('regular configured cadence required')
    out = {}
    for k in keys:
        # No backfill. Missing initial values remain NaN until training-fitted imputation.
        v = observations[k].astype(float)
        if k == 'emitted_intensity': v = v/observations.nominal_intensity
        out[k+'__value'] = v
        # Instantaneous ablation deliberately excludes freshness/history flags.
        if temporal:
            for h in ('age_s','missing','saturated','stale'):
                out[k+'__'+h] = observations[k+'__'+h]
        if temporal:
            out[k+'__delta'] = v.diff()
            for seconds in c['windows_s']:
                w=max(2,int(round(seconds/dt))); r=v.rolling(w,min_periods=2)
                out[f'{k}__mean_{seconds}'] = r.mean()
                out[f'{k}__std_{seconds}'] = r.std()
                out[f'{k}__iqr_{seconds}'] = r.quantile(.75)-r.quantile(.25)
                out[f'{k}__slope_{seconds}'] = (v-v.shift(w-1))/((w-1)*dt)
            lo,hi=c['engineering_limits'][k]
            raw=observations[k]
            outside=(raw<lo)|(raw>hi)
            groups=(~outside).cumsum()
            out[k+'__outside_s']=outside.groupby(groups).cumsum()*dt
    if environment:
        out['intensity_temperature_residual']=observations.emitted_intensity/observations.nominal_intensity-1-c['assumptions']['intensity_temperature']*(observations.temperature-c['assumptions']['device_temperature'][0])
        out['bias_temperature_residual']=observations.modulator_bias_v-c['assumptions']['phase_temperature']*(observations.temperature-c['assumptions']['device_temperature'][0])
    return pd.DataFrame(out,index=observations.index).replace([np.inf,-np.inf],np.nan)
