"""Immutable Phase 3 reproduction: --quick or --bounded --seeds 42 43.

Every run is exploratory synthetic evidence, not a frozen hardware benchmark.
Only --activate-local changes the local dashboard's manifest pointer.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from src.config import git_commit
from src.data.source_side import generate_source_sessions
from src.data.source_artifacts import write_json, save_sessions, complete_run, CORE, digest
from src.models.source_side import SourceMonitor
from src.simulator.source_side import load_source_config, SourceSideSimulator
from src.evaluation.source_side import event_metrics, summarize_events, cluster_intervals
from src.evaluation.metrics import expected_calibration_error


def evaluate(model, sessions, seed, variant, regime):
    rows, targets, predictions, probabilities, rejection = [], [], [], [], []
    for session in sessions:
        scores = model.raw_scores(session)
        for detector, cutoff in model.thresholds.items():
            alarm, _ = model.policy(scores[detector], cutoff)
            rows.append(dict(seed=seed, variant=variant, regime=regime, detector=detector,
                             **event_metrics(session, alarm)))
        targets.extend(session.truth.family.tolist())
        predictions.extend(model.classes[scores['probabilities'].argmax(1)].tolist())
        probabilities.extend(scores['probabilities'])
        rejection.extend(model.predict(session)['uncertain'])
    y, p, reject = np.array(targets), np.array(probabilities), np.array(rejection)
    known = np.isin(y, model.classes)
    indices = np.searchsorted(model.classes, y[known])
    onehot = np.eye(len(model.classes))[indices]
    report = classification_report(y,predictions,labels=list(range(5)),output_dict=True,zero_division=0)
    report['calibration'] = dict(known_samples=int(known.sum()),
        ece=expected_calibration_error(indices,p[known]),
        multiclass_brier=float(np.mean(np.sum((p[known]-onehot)**2,axis=1))),
        log_loss=float(-np.log(np.maximum(p[known][np.arange(known.sum()),indices],1e-12)).mean()))
    report['rejection'] = dict(known_false_rejection=float(reject[known].mean()),
        unknown_samples=int((~known).sum()),
        unknown_rejection=float(reject[~known].mean()) if (~known).any() else None,
        confidence_cutoff=model.unknown_cutoff)
    return rows, report


def extended_benign(seed, ids, profile, config, seconds):
    c = copy.deepcopy(config)
    c['samples'] = int(seconds/c['dt_s'])
    return [SourceSideSimulator(c).simulate(seed,d,1,0,profile) for d in ids]


def shortcut_audit(train, test, model):
    """No truth in features; pre-onset matched nuisances must not predict scenario."""
    from sklearn.ensemble import RandomForestClassifier
    def data(sessions):
        x, y = [], []
        for s in sessions:
            if s.metadata['session_id'] != 0:
                continue
            # One common pre-onset sample per paired device avoids row weighting.
            x.append(model.features(s).iloc[0].fillna(0).to_numpy())
            y.append(s.metadata['scenario'])
        return np.array(x), np.array(y)
    x,y = data(train); xt,yt = data(test)
    clf = RandomForestClassifier(n_estimators=30,random_state=0).fit(x,y)
    features = model.feature_names
    assert not any(any(word in name for word in ('scenario','damage_db','seed_state','internal_power')) for name in features)
    # Attack classes 0..4 only: unseen chain not part of shortcut classifier score.
    mask = yt < 5
    return dict(preonset_scenario_accuracy=float((clf.predict(xt[mask])==yt[mask]).mean()),
        balanced_chance=.2, forbidden_feature_audit='passed',
        train_test_devices_disjoint=not bool(set(s.metadata['device_id'] for s in train)&set(s.metadata['device_id'] for s in test)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,43,44,45,46])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--activate-local', action='store_true')
    args = parser.parse_args()
    initial_hashes={name:digest(ROOT/name) for name in CORE}
    cfg = load_source_config()
    seeds = [1] if args.quick else args.seeds
    mode = 'smoke' if args.quick else 'bounded' if args.bounded else 'full'
    if args.quick:
        cfg.update(samples=100, windows_s=[10,20,40])
        cfg['monitor']['model_estimators'] = 20
    elif args.bounded:
        cfg['monitor']['model_estimators'] = 40
    counts = (4,2,2) if args.quick else (6,4,3) if args.bounded else (12,6,6)
    ids = [list(range(counts[0])),list(range(100,100+counts[1])),list(range(200,200+counts[2]))]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out = ROOT/'results/phase3/runs'/stamp
    out.mkdir(parents=True, exist_ok=False)
    data = out/'data'; data.mkdir()
    rows, reports, operating, schemas, deployment, audits, evaluations = [], {}, {}, {}, {}, {}, []
    start = time.time()
    for seed in seeds:
        for profile in ('minimal','instrumented'):
            train, validation = [generate_source_sessions(seed,i,range(5),profile,cfg,args.workers) for i in ids[:2]]
            test = generate_source_sessions(seed,ids[2],range(6),profile,cfg,args.workers)
            duration = 400 if args.quick else 3600
            validation += extended_benign(seed,ids[1],profile,cfg,duration)
            test += extended_benign(seed,ids[2],profile,cfg,duration)
            for split,sessions in [('train',train),('validation',validation),('nominal',test)]:
                save_sessions(data,f'{profile}_{seed}_{split}',sessions)
            model = SourceMonitor(profile,seed=seed,config=cfg).fit(train,validation)
            filename = f'data/monitor_{profile}_{seed}.joblib'
            joblib.dump(model,out/filename)
            schemas[filename] = model.feature_names
            if seed==seeds[0]:
                deployment[profile] = filename  # declared choice, never glob ordering
            key = f'{profile}_{seed}'
            operating[key] = dict(thresholds=model.thresholds,validation=model.validation_operating_points,
                calibration_devices=model.calibration_devices,selection_devices=model.selection_devices,
                temperature=model.temperature)
            audits[key] = shortcut_audit(train,test,model)
            regimes = {'nominal':test}
            stresses = ('weak','startup') if args.quick else ('weak','startup','slow','intermittent','sensor_failure','offset','environment','spectral_blind')
            for stress in stresses:
                regimes[stress] = generate_source_sessions(seed,ids[2],range(6),profile,cfg,args.workers,stress)
                save_sessions(data,f'{profile}_{seed}_{stress}',regimes[stress])
            for regime,sessions in regimes.items():
                rr,rp = evaluate(model,sessions,seed,profile,regime)
                rows += rr; reports[f'{key}_{regime}'] = rp
                evaluations.append(dict(model=filename,sessions=f'{profile}_{seed}_{regime}',seed=seed,variant=profile,regime=regime))
            print(key, summarize_events([r for r in rows if r['seed']==seed and r['variant']==profile and r['regime']=='nominal' and r['detector']=='combined']),flush=True)
            if not args.quick:
                for name,temporal,environment,held in [('instantaneous',False,True,None),('no_environment',True,False,None),
                        ('heldout_photo',True,True,3),('heldout_seeding',True,True,2)]:
                    tr = [s for s in train if held is None or s.metadata['scenario']!=held]
                    va = [s for s in validation if held is None or s.metadata['scenario']!=held]
                    # Cross-effects can create held-out labels in other mechanisms.
                    # Remove those whole sessions, never relabel or drop selected rows.
                    if held is not None:
                        tr = [s for s in tr if held not in set(s.truth.family)]
                        va = [s for s in va if held not in set(s.truth.family)]
                    m = SourceMonitor(profile,temporal,environment,seed,cfg).fit(tr,va)
                    variant = profile+'_'+name
                    fn = f'data/monitor_{variant}_{seed}.joblib'
                    joblib.dump(m,out/fn); schemas[fn] = m.feature_names
                    rr,rp = evaluate(m,test,seed,variant,'nominal')
                    rows += rr; reports[f'{variant}_{seed}'] = rp
                    evaluations.append(dict(model=fn,sessions=f'{profile}_{seed}_nominal',seed=seed,variant=variant,regime='nominal'))
    pd.DataFrame(rows).to_csv(out/'events.csv',index=False)
    write_json(out/'events.json',rows)
    summaries, intervals = [], []
    keys = sorted({(r['variant'],r['regime'],r['detector']) for r in rows})
    for variant,regime,detector in keys:
        group = [r for r in rows if (r['variant'],r['regime'],r['detector'])==(variant,regime,detector)]
        summaries.append(dict(variant=variant,regime=regime,detector=detector,**summarize_events(group)))
        if detector in ('combined','receiver'):
            intervals.append(dict(variant=variant,regime=regime,detector=detector,intervals=cluster_intervals(group)))
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    for name,value in [('classification',reports),('operating_points',operating),('shortcut_audit',audits),('cluster_intervals',intervals)]:
        write_json(out/(name+'.json'),value)
    manifest = dict(mode=mode,seeds=seeds,config=cfg,device_partitions=dict(zip(('train','validation','test'),ids)),
        config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),git_commit=git_commit(),
        seconds=time.time()-start,provenance='SIMULATED DATA; exploratory audit; no hardware validation',
        command=sys.argv,feature_schemas=schemas,deployment=deployment,evaluations=evaluations,
        semantics='v2 new alarm during exposure; startup scores from zero; separate persistent effects',
        prior_results='results/phase3/full are superseded prototype outputs; preserved unchanged')
    if any(digest(ROOT/name)!=value for name,value in initial_hashes.items()):
        raise RuntimeError('Source changed during run; incomplete run will not be activated')
    complete_run(out,manifest)
    if args.activate_local:
        if args.quick:
            raise ValueError('smoke models cannot be activated')
        pointer = ROOT/'results/phase3/active.json'
        temp = pointer.with_suffix('.tmp')
        write_json(temp,dict(run=out.relative_to(ROOT).as_posix()))
        temp.replace(pointer)
    print('COMPLETED RUN: '+str(out),flush=True)


if __name__=='__main__':
    main()
