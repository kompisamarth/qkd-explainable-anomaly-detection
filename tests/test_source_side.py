"""Phase 3 physical invariants, causality, group separation and privacy."""
import copy
import numpy as np
import pandas as pd
import pytest
from src.simulator.source_side import SourceSideSimulator, AttackSchedule, load_source_config
from src.features.source_side import extract_source_features
from src.data.source_side import generate_source_sessions
from src.evaluation.source_side import alarm_policy,event_metrics
from src.models.source_side import SourceMonitor


@pytest.fixture
def config():
    c=load_source_config();c['samples']=100;c['monitor']['model_estimators']=10
    return c


def test_reproducibility_workers_and_counterfactual(config):
    a=generate_source_sessions(2,[1],[0,1],config=config,workers=1)
    b=generate_source_sessions(2,[1],[0,1],config=config,workers=2)
    for x,y in zip(a,b): pd.testing.assert_frame_equal(x.observations,y.observations)
    pd.testing.assert_series_equal(a[0].observations.temperature,a[1].observations.temperature)
    assert not a[0].truth.active.any()
    assert a[0].truth.damage_db.max()==0
    assert a[0].truth.photo_state.max()==0


def test_photo_recovery_damage_persistence_chain(config):
    sim=SourceSideSimulator(config)
    photo=sim.simulate(schedule=AttackSchedule(3,20,80,60,10000,800),scenario=3)
    assert photo.truth.photo_state.iloc[39]>photo.truth.photo_state.iloc[-1]>0
    damage=sim.simulate(schedule=AttackSchedule(4,20,80,60,1000000,1550),scenario=4)
    assert damage.truth.damage_db.iloc[-1]==damage.truth.damage_db.iloc[40]>0
    assert damage.truth.family.iloc[-1]==4
    chain=sim.simulate(schedule=AttackSchedule(5,20,160,100,1000000,1550),scenario=5)
    assert set(chain.truth.family[10:50])=={4}
    assert set(chain.truth.family[50:80])=={1}
    assert chain.truth.damage_db.iloc[50]>0
    repair=sim.simulate(schedule=AttackSchedule(4,20,80,60,1000000,1550,repair_s=100),scenario=4)
    assert repair.truth.damage_db.iloc[-1]==0


def test_sensor_freshness_saturation_and_missing(config):
    c=copy.deepcopy(config);c['missing_probability']=1
    s=SourceSideSimulator(c).simulate(profile='instrumented')
    assert s.observations.isolation_db.isna().all()
    assert (s.observations.isolation_db__age_s.diff().dropna()>0).all()
    c['missing_probability']=0
    s=SourceSideSimulator(c).simulate(scenario=1,profile='instrumented',schedule=AttackSchedule(1,0,200,100,1e8,1550))
    assert s.observations.watchdog_power.max()<=100
    assert s.observations.watchdog_power__saturated.any()
    assert s.observations.isolation_db.iloc[1]==s.observations.isolation_db.iloc[0]
    assert s.observations.isolation_db__age_s.iloc[1]==2


def test_causal_features_and_ground_truth_allowlist(config):
    s=SourceSideSimulator(config).simulate(scenario=3)
    before=extract_source_features(s.observations,config=config)
    obs=s.observations.copy();obs.loc[60:,'emitted_intensity']=100
    obs['scenario']=5;obs['damage_db']=999
    after=extract_source_features(obs,config=config)
    pd.testing.assert_frame_equal(before.iloc[:60],after.iloc[:60])
    assert list(before)==list(after)
    assert not any('scenario' in c or 'damage_db' in c for c in before)


def test_alarm_event_counts_and_missing_latency(config):
    a,w=alarm_policy([0,2,2,2,0,0,0,0,0],1,3,5)
    assert a.tolist()==[False,False,False,True,True,True,True,True,False]
    benign=SourceSideSimulator(config).simulate()
    metric=event_metrics(benign,np.ones(100,bool))
    assert metric['false_alarm_events']==0  # no new edge at warm-up
    attack=SourceSideSimulator(config).simulate(scenario=1,stress='startup')
    missed=event_metrics(attack,np.zeros(100,bool))
    assert missed['detected'] is False and missed['latency_s'] is None


@pytest.fixture(scope='module')
def trained():
    c=load_source_config();c['samples']=90;c['monitor']['model_estimators']=10
    tr=generate_source_sessions(5,[1,2],range(5),config=c)
    va=generate_source_sessions(5,[10,11],range(5),config=c)
    return SourceMonitor(config=c).fit(tr,va),tr,va


def test_train_only_fit_and_schema(trained):
    m,tr,va=trained
    assert not set(m.train_devices)&set(m.validation_devices)
    with pytest.raises(ValueError,match='leakage'): SourceMonitor().fit(tr,tr)
    medians=m.imputer.statistics_.copy(); threshold=copy.deepcopy(m.thresholds)
    altered=copy.deepcopy(va[0]);altered.observations['emitted_intensity']=100
    m.predict(altered)
    np.testing.assert_array_equal(medians,m.imputer.statistics_)
    assert threshold==m.thresholds
    m.feature_names.append('invalid')
    with pytest.raises(ValueError,match='schema'):m.predict(va[0])
    m.feature_names.pop()


def test_blind_api_no_disclosure_and_expiry(trained,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend import source_side as api
    m,_,_=trained
    monkeypatch.setattr(api,'load_monitor',lambda profile:(m,'test.joblib'))
    app=FastAPI();app.include_router(api.router);client=TestClient(app)
    response=client.post('/api/source/simulate',json={'blind':True}).json()
    assert 'truth' not in response
    assert 'seed' not in response['provenance']
    assert not {'scenario','start_s','damage_db','internal_power_nw'} & set(response['observations'][0])
    sid=response['session_id']
    explained=client.get(f'/api/source/explain/{sid}?index=80')
    assert explained.status_code==200
    assert 'truth' not in explained.json()
    truth=client.post('/api/source/reveal/'+sid)
    assert truth.status_code==200 and 'scenario' in truth.json()['metadata']
    assert client.get('/api/source/explain/expired').status_code==404


def test_protocol_adapter_is_observable_not_security_bound():
    from src.simulator.source_protocol import bb84_source_observables,mdi_source_observables
    assert bb84_source_observables(.8,0)[0]>bb84_source_observables(.5,0)[0]
    out=mdi_source_observables(.5,.6)
    assert 0<=out['qber']<=1 and 'key_rate' not in out
