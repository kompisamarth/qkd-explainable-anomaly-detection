"""Versioned, checksummed Phase 3 artifacts; validate before loading pickle."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import joblib
import pandas as pd
from src.simulator.source_side import ROOT, SourceSession

SCHEMA = 2
PACKAGES = ('numpy','scipy','pandas','scikit-learn','xgboost','shap','joblib','pyarrow')
CORE = ('src/simulator/source_side.py','src/simulator/source_protocol.py',
        'src/features/source_side.py','src/models/source_side.py','src/evaluation/source_side.py',
        'config/source_side.yaml')
CORE += ('src/simulator/calibrated.py','src/simulator/mdi_qkd.py','src/simulator/tf_qkd.py',
         'src/config.py','src/evaluation/metrics.py','src/data/source_side.py',
         'src/data/source_artifacts.py','scripts/13_source_side_pipeline.py',
         'scripts/14_source_monitor_replay.py','requirements-phase3-lock.txt')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def versions():
    return dict(python=platform.python_version(), **{p:importlib.metadata.version(p) for p in PACKAGES})


def save_sessions(directory, name, sessions):
    directory = Path(directory)
    obs, truth, metadata = [], [], []
    for index, s in enumerate(sessions):
        obs.append(s.observations.assign(artifact_session=index))
        truth.append(s.truth.assign(artifact_session=index))
        metadata.append(s.metadata)
    pd.concat(obs).to_parquet(directory / (name+'_observations.parquet'), index=False)
    pd.concat(truth).to_parquet(directory / (name+'_truth.parquet'), index=False)
    write_json(directory / (name+'_metadata.json'), metadata)


def load_sessions(directory, name):
    directory = Path(directory)
    obs = pd.read_parquet(directory / (name+'_observations.parquet'))
    truth = pd.read_parquet(directory / (name+'_truth.parquet'))
    metadata = json.loads((directory / (name+'_metadata.json')).read_text())
    return [SourceSession(obs[obs.artifact_session==i].drop(columns='artifact_session').reset_index(drop=True),
                          truth[truth.artifact_session==i].drop(columns='artifact_session').reset_index(drop=True), m)
            for i,m in enumerate(metadata)]


def complete_run(directory, manifest):
    directory = Path(directory)
    for name in CORE:
        target = directory/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    manifest.update(schema=SCHEMA, status='complete', versions=versions(),
                    source_hashes={name:digest(ROOT/name) for name in CORE})
    manifest['checksums'] = {p.relative_to(directory).as_posix():digest(p)
                            for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write_json(directory/'manifest.json',manifest)


def validate_run(directory, current_source=True):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('status')!='complete' or manifest.get('schema')!=SCHEMA:
        raise ValueError('incomplete or incompatible run')
    for name, expected in manifest['checksums'].items():
        target = (directory/name).resolve()
        if not target.is_relative_to(directory) or not target.is_file() or digest(target)!=expected:
            raise ValueError('corrupt artifact: '+name)
    if current_source and any(digest(ROOT/name)!=value for name,value in manifest['source_hashes'].items()):
        raise ValueError('stale source checkpoint; reproduce after source changes')
    if manifest['versions']!=versions():
        raise ValueError('checkpoint runtime differs from recorded pinned environment')
    return manifest


def load_model(directory, filename, manifest):
    if filename not in manifest['checksums']:
        raise ValueError('model not listed in manifest')
    model = joblib.load(Path(directory)/filename)
    if model.schema_version!=SCHEMA or model.feature_names!=manifest['feature_schemas'][filename]:
        raise ValueError('checkpoint feature schema mismatch')
    return model
