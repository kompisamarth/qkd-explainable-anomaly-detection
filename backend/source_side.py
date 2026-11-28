"""Source monitor API: blind truth stays server-side until explicit reveal."""
from collections import OrderedDict
import json
import secrets
import threading
import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.simulator.source_side import ROOT, SourceSideSimulator, SCENARIOS, FAMILIES
from src.data.source_artifacts import validate_run,load_model,digest
from src.evaluation.source_side import event_metrics

router=APIRouter(prefix='/api/source',tags=['Source monitor'])
sessions=OrderedDict(); lock=threading.Lock()
model_cache={}


def active_run():
    pointer=ROOT/'results/phase3/active.json'
    if not pointer.exists():
        raise ValueError('No active source run. Reproduce with --activate-local.')
    directory=(ROOT/json.loads(pointer.read_text())['run']).resolve()
    if not directory.is_relative_to((ROOT/'results/phase3/runs').resolve()):
        raise ValueError('Invalid active run path')
    return directory


def load_monitor(profile):
    try:
        directory=active_run()
        # Check source, runtime and checksums on access; cache only deserialization.
        manifest=validate_run(directory)
        filename=manifest['deployment'][profile]
        key=(str(directory),filename,digest(directory/filename))
        with lock:
            if key not in model_cache:
                if len(model_cache)>=4:
                    model_cache.clear()
                model_cache[key]=load_model(directory,filename,manifest)
            model=model_cache[key]
        if model.profile!=profile:
            raise ValueError('checkpoint profile mismatch')
        return model,filename
    except (ValueError,KeyError,OSError) as exc:
        raise HTTPException(503,str(exc)) from exc


class SourceRequest(BaseModel):
    profile: str='minimal'
    scenario: int=Field(default=1,ge=0,le=5)
    seed: int=Field(default=42,ge=0,le=1000000000)
    blind: bool=True
    stress: str='nominal'


def prediction_with_origin(model, session):
    prediction=model.predict(session)
    origins=[]; origin=None; previous=False
    for alarm,branch in zip(prediction['alarm'],prediction['responsible_branch']):
        if alarm and not previous:
            origin=branch
        if not alarm:
            origin=None
        origins.append(origin)
        previous=alarm
    prediction['alarm_origin']=origins
    return prediction


@router.get('/meta')
def meta():
    available={}; errors={}
    for profile in ('minimal','instrumented'):
        try:
            load_monitor(profile); available[profile]=True
        except HTTPException as exc:
            available[profile]=False; errors[profile]=exc.detail
    return dict(scenarios=SCENARIOS,families=FAMILIES,profiles=['minimal','instrumented'],
                model_available=available,errors=errors,provenance='SIMULATED DATA')


@router.post('/simulate')
def simulate(req:SourceRequest):
    if req.profile not in ('minimal','instrumented'): raise HTTPException(400,'Unknown profile')
    if req.stress not in ('nominal','weak','startup','slow','intermittent','sensor_failure','offset','environment','spectral_blind'):
        raise HTTPException(400,'Unknown stress')
    m,filename=load_monitor(req.profile)
    seed=secrets.randbelow(1000000000) if req.blind else req.seed
    scenario=secrets.randbelow(6) if req.blind else req.scenario
    s=SourceSideSimulator(m.config).simulate(seed,9999,0,scenario,req.profile,stress=req.stress)
    prediction=prediction_with_origin(m,s); sid=secrets.token_urlsafe(24)
    # Recover training-benign residual reference bands in sensor units.
    bands={}
    for i in m.value_indices:
        name=m.feature_names[i].split('__')[0]
        middle=m.centres[i]*m.scaler.scale_[i]+m.scaler.center_[i]
        half=m.scales[i]*m.scaler.scale_[i]
        bands[name]=[float(middle-half),float(middle+half)]
    with lock:
        sessions[sid]=(s,m)
        while len(sessions)>32: sessions.popitem(last=False)
    result=dict(session_id=sid,blind=req.blind,observations=json.loads(s.observations.to_json(orient='records')),
                prediction=prediction,thresholds=m.thresholds,limits=m.config['engineering_limits'],bands=bands,
                units={k:v[0] for k,v in m.config['sensors'].items()},
                provenance=dict(label='SIMULATED DATA',model=filename,assumptions='docs/PHASE3.md'))
    if not req.blind: result['truth']=reveal(sid)
    return result


def lookup(sid):
    with lock: value=sessions.get(sid)
    if value is None: raise HTTPException(404,'Source session expired; rerun simulation')
    return value


@router.post('/reveal/{sid}')
def reveal(sid:str):
    s,m=lookup(sid)
    return dict(metadata=s.metadata,truth=json.loads(s.truth.to_json(orient='records')),
                metrics=event_metrics(s,m.predict(s)['alarm']))


@router.get('/explain/{sid}')
def explain(sid:str,index:int=0):
    s,m=lookup(sid)
    if not 0<=index<len(s.observations): raise HTTPException(400,'Sample index outside session')
    prediction=prediction_with_origin(m,s)
    return dict(attributions=m.explain(s,index),responsible_branch=prediction['responsible_branch'][index],
                alarm_origin=prediction['alarm_origin'][index],
                note='Classifier SHAP: raw class-margin attribution, not anomaly/fusion attribution or proof of physical causation.',provenance='SIMULATED DATA')


@router.get('/benchmarks')
def benchmarks():
    import pandas as pd
    try:
        directory=active_run(); validate_run(directory)
    except (ValueError,KeyError,OSError) as exc:
        raise HTTPException(503,str(exc)) from exc
    p=directory/'summary.csv'
    if not p.exists(): raise HTTPException(404,'Phase 3 evaluation not generated')
    strict=p.with_name('onset_summary.csv')
    return dict(rows=json.loads(pd.read_csv(p).to_json(orient='records')),
                new_alarm_summary=json.loads(pd.read_csv(strict).to_json(orient='records')) if strict.exists() else [],
                provenance='SIMULATED DATA')
