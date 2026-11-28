"""One event evaluator for validation, tests, replay and dashboard.

New alarm edges during exposure earn detection. Warm-up never creates an edge.
Startup attacks score from zero; other sessions score after 60 s. Intermittent
pulses belong to one scheduled episode. Chain stages require independent edges.
Persistent effects and left/right censoring are reported separately.
"""
import numpy as np


def alarm_policy(score, threshold, persistence=3, clear_samples=5):
    state = False
    high = low = 0
    alarms, warnings = [], []
    for value in score:
        high = high + 1 if np.isfinite(value) and value > threshold else 0
        low = low + 1 if not np.isfinite(value) or value <= threshold * .8 else 0
        if high >= persistence:
            state = True
        if low >= clear_samples:
            state = False
        alarms.append(state)
        warnings.append(bool(np.isfinite(value) and value > threshold * .8))
    return np.asarray(alarms), np.asarray(warnings)


def event_metrics(session, alarm, warmup_s=60):
    t = session.observations.time_s.to_numpy()
    dt = session.metadata['dt_s']
    alarm = np.asarray(alarm, bool)
    if len(alarm) != len(t):
        raise ValueError('alarm/session length mismatch')
    meta = session.metadata
    benign = meta['scenario'] == 0
    onset, stop = meta['start_s'], meta['stop_s']
    startup = bool(not benign and onset <= t[0])
    valid = t >= (t[0] if startup else warmup_s)
    starts = alarm & ~np.r_[False, alarm[:-1]] & valid
    active = session.truth.active.to_numpy(bool)
    degraded = session.truth.degraded.to_numpy(bool)
    pre = valid & (t < onset) if not benign else np.zeros(len(t), bool)
    hits = np.flatnonzero(starts & active & (t >= onset))
    effects = np.flatnonzero(starts & ~active & degraded & (t >= onset))
    second = meta['second_s']
    chain = meta['scenario'] == 5
    return dict(device_id=meta['device_id'], session_id=meta.get('session_id', 0),
        scenario=meta['scenario'], stress=meta['stress'],
        benign_hours=float(valid.sum()*dt/3600) if benign else 0.,
        false_alarm_events=int(starts.sum()) if benign else 0,
        preonset_hours=float(pre.sum()*dt/3600),
        preonset_false_alarm_events=int((starts & pre).sum()),
        detected=bool(len(hits)) if not benign else None,
        latency_s=float(t[hits[0]]-onset) if len(hits) and not benign else None,
        persistent_effect_detected=bool(len(effects)) if not benign else None,
        effect_latency_s=float(t[effects[0]]-stop) if len(effects) and t[effects[0]] >= stop else None,
        alarm_fraction=float(alarm[valid].mean()) if valid.any() else None,
        precursor_detected=bool(np.any(starts & active & (t < second))) if chain else None,
        second_stage_detected=bool(np.any(starts & active & (t >= second))) if chain else None,
        startup=startup, left_censored=bool(not benign and not startup and onset < warmup_s),
        right_censored=bool(not benign and stop > t[-1]+dt),
        effect_right_censored=bool(degraded[-1]))


def summarize_events(rows):
    def present(key):
        return [r[key] for r in rows if r.get(key) is not None]
    hours = sum(r['benign_hours'] for r in rows)
    fa = sum(r['false_alarm_events'] for r in rows)
    ph = sum(r.get('preonset_hours', 0) for r in rows)
    pf = sum(r.get('preonset_false_alarm_events', 0) for r in rows)
    attacks, lat = present('detected'), present('latency_s')
    def average(key):
        v = present(key)
        return float(np.mean(v)) if v else None
    return dict(benign_hours=hours, false_alarm_events=fa,
        false_alarms_per_hour=fa/hours if hours else None,
        preonset_hours=ph, preonset_false_alarm_events=pf,
        preonset_false_alarms_per_hour=pf/ph if ph else None,
        attack_events=len(attacks), detection_rate=float(np.mean(attacks)) if attacks else None,
        missed_events=sum(not x for x in attacks),
        persistent_effect_detection_rate=average('persistent_effect_detected'),
        precursor_detection_rate=average('precursor_detected'),
        second_stage_detection_rate=average('second_stage_detected'),
        detected_latency_mean_s=float(np.mean(lat)) if lat else None)


def cluster_intervals(rows, draws=500, seed=2026):
    """Seed/model resampling then device clusters, retaining paired scenarios.

    Single-seed intervals are conditional on that model. Zero-event bootstrap
    FAR intervals are not an upper confidence guarantee.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted({r['seed'] for r in rows})
    clusters = {s: {} for s in seeds}
    for row in rows:
        clusters[row['seed']].setdefault(row['device_id'], []).append(row)
    keys = ['detection_rate', 'false_alarms_per_hour', 'detected_latency_mean_s',
            'precursor_detection_rate', 'second_stage_detection_rate']
    values = {k: [] for k in keys}
    for _ in range(draws):
        sample = []
        for s in rng.choice(seeds, len(seeds)):
            devices = list(clusters[s])
            for d in rng.choice(devices, len(devices)):
                sample.extend(clusters[s][d])
        summary = summarize_events(sample)
        for key in keys:
            if summary[key] is not None:
                values[key].append(summary[key])
    return {k: dict(low=float(np.quantile(v, .025)), high=float(np.quantile(v, .975)),
                   valid_draws=len(v), seeds=len(seeds)) if v else None for k, v in values.items()}
