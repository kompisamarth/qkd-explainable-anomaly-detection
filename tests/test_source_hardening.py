"""Regression tests for the Phase 3 audit, independent of reported scores."""
import copy
import numpy as np
import pandas as pd
import pytest
from src.simulator.source_side import SourceSideSimulator, AttackSchedule, load_source_config
from src.features.source_side import extract_source_features
from src.evaluation.source_side import event_metrics


def case(scenario=1):
    c = load_source_config()
    c['samples'] = 100
    return SourceSideSimulator(c).simulate(scenario=scenario,
        schedule=AttackSchedule(scenario, 80, 140, 110, 1000, 1550))


def test_alarm_crossing_onset_is_not_a_detection():
    s = case()
    a = np.arange(100) >= 20
    r = event_metrics(s, a)
    assert not r['detected'] and r['latency_s'] is None
    assert r['preonset_false_alarm_events'] == 0  # transition occurred before warm-up


def test_warmup_does_not_create_alarm_transition():
    s = case(0)
    assert event_metrics(s, np.ones(100, bool))['false_alarm_events'] == 0


def test_chain_requires_new_alarm_for_second_stage():
    s = case(5)
    a = (np.arange(100) >= 45) & (np.arange(100) < 65)
    r = event_metrics(s, a)
    assert r['precursor_detected']
    assert not r['second_stage_detected']


def test_shared_sensors_are_identical_across_profiles():
    a = SourceSideSimulator().simulate(profile='minimal')
    b = SourceSideSimulator().simulate(profile='instrumented')
    pd.testing.assert_frame_equal(a.observations, b.observations[a.observations.columns])


def test_held_saturation_and_staleness():
    c = load_source_config()
    c['missing_probability'] = 0
    c['sensors']['isolation_db'][3] = 1
    s = SourceSideSimulator(c).simulate(profile='instrumented')
    assert s.observations.isolation_db__saturated.iloc[1] == 1
    c['missing_probability'] = 1
    s = SourceSideSimulator(c).simulate()
    assert s.observations.watchdog_power__stale.iloc[-1] == 1


def test_irregular_cadence_rejected():
    s = case()
    s.observations.loc[5, 'time_s'] += .5
    with pytest.raises(ValueError, match='cadence'):
        extract_source_features(s.observations)


def test_schedule_and_config_validation():
    with pytest.raises(ValueError):
        SourceSideSimulator().simulate(scenario=1, schedule=AttackSchedule(2, 20, 10, 15, -1, 1550))
    c = load_source_config()
    c['missing_probability'] = 2
    with pytest.raises(ValueError):
        SourceSideSimulator(c)


def test_missed_phase2_latency_is_missing():
    from src.evaluation.metrics import detection_latency
    frame = pd.DataFrame(dict(session_id=[0]*4, block_index=[0,1,2,3],
        attack_start_block=[2]*4, session_attack_class=[1]*4, session_p_eve=[.2]*4))
    assert pd.isna(detection_latency(frame, np.zeros(4, bool)).latency_blocks.iloc[0])


def test_blinding_trigger_is_shared_across_intensities():
    from tests.test_calibrated_physics import fixed_session, quiet_physics
    from src.simulator.calibrated import CalibratedBB84Simulator
    sp=fixed_session('detector_blinding',intensity=1,residual_qber=.005,gain_mismatch=0)
    sp.intensities=np.array([.5,.1,0.])
    sp.intensity_probs=np.array([.8,.15,.05])
    stats=CalibratedBB84Simulator(quiet_physics())._cell_stats(sp,.1,.015,sp.eta,True)
    gain=stats['det_match'][:,0]
    assert gain[0]/(1-np.exp(-.5)) == pytest.approx(gain[1]/(1-np.exp(-.1)))
    assert gain[2]==0


def test_bb84_blind_response_seals_truth(monkeypatch):
    from fastapi.testclient import TestClient
    from backend import server
    class Physics:
        intensity=.2
    monkeypatch.setattr(server,'_simulate',lambda *args,**kw:(None,Physics()))
    monkeypatch.setattr(server,'_package',lambda *args,**kw:dict(session_id='test',class_id=1,
        class_name='IR',class_full='IR',p_eve=.2,onset=76,summary={'post_accuracy':1,'delta_t':3},
        provenance={'seed':42},pred=[1],blocks=[76]))
    client=TestClient(server.app)
    response=client.post('/api/simulate/blind',json={}).json()
    assert not {'class_id','class_name','p_eve','onset'} & response.keys()
    assert 'post_accuracy' not in response['summary']
    assert 'seed' not in response['provenance']
    assert client.post('/api/simulate/reveal/test').json()['class_id']==1


def test_effect_detection_is_separate_and_cessation_is_exclusive():
    s=case()
    s.truth.loc[70:,'degraded']=1
    alarm=np.arange(100)>=70  # exactly stop_s, outside active exposure
    result=event_metrics(s,alarm)
    assert not result['detected'] and result['latency_s'] is None
    assert result['persistent_effect_detected'] and result['effect_latency_s']==0


def test_startup_policy_and_new_chain_stage():
    s=case(5)
    s.metadata['start_s']=0
    s.truth.loc[:54,'active']=1
    a=np.zeros(100,bool);a[2:5]=True;a[58:63]=True
    r=event_metrics(s,a)
    assert r['startup'] and r['latency_s']==4
    assert r['precursor_detected'] and r['second_stage_detected']


def test_zero_click_qber_and_cadence_scaled_counts(monkeypatch):
    import src.simulator.source_side as module
    monkeypatch.setattr(module,'bb84_source_observables',lambda *a,**k:(0.,0.))
    c=load_source_config();c['dt_s']=1
    s=SourceSideSimulator(c).simulate()
    assert s.observations.protocol_qber.isna().all()
    assert (s.observations.protocol_detections==0).all()
    assert (s.observations.protocol_pulses==50000).all()


def test_instantaneous_features_have_no_history_flags():
    features=extract_source_features(case().observations,temporal=False)
    assert not any(any(token in col for token in ('age_s','delta','outside','mean_','stale')) for col in features)


def test_fusion_excursions_do_not_saturate_and_are_scale_invariant():
    from src.models.source_side import SourceMonitor
    ref=np.arange(100,dtype=float)
    values=np.array([120.,150.])
    scores=SourceMonitor.normalized_excursion(values,ref)
    assert scores[1]>scores[0]>1
    np.testing.assert_allclose(scores,SourceMonitor.normalized_excursion(values*2+5,ref*2+5))


def test_artifact_corruption_and_incomplete_detection(tmp_path):
    from src.data.source_artifacts import complete_run,validate_run
    with pytest.raises(FileNotFoundError):validate_run(tmp_path)
    payload=tmp_path/'payload.txt';payload.write_text('original')
    complete_run(tmp_path,{})
    assert validate_run(tmp_path)['status']=='complete'
    payload.write_text('corrupt')
    with pytest.raises(ValueError,match='corrupt'):validate_run(tmp_path)


def test_missing_active_model_returns_503(monkeypatch):
    from backend import source_side as api
    from fastapi import HTTPException
    monkeypatch.setattr(api,'active_run',lambda: (_ for _ in ()).throw(ValueError('missing model')))
    with pytest.raises(HTTPException) as error:api.load_monitor('minimal')
    assert error.value.status_code==503
