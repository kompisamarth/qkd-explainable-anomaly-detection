"""Stateful source + causal sensor simulator. Numerical choices are assumptions.

Measurements and hidden truth are separate frames. Never train on truth columns.
The model is an effective component model, not hardware validation or a proof.
"""
from dataclasses import dataclass, field, asdict
import hashlib
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import yaml
from src.simulator.source_protocol import bb84_source_observables

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ['Benign', 'Trojan horse', 'Laser seeding', 'Photorefraction',
             'Laser damage', 'Damage then Trojan horse']
FAMILIES = SCENARIOS[:5]  # chains are evaluated as sequences, never future labels
MINIMAL = ['watchdog_power', 'emitted_intensity', 'temperature', 'modulator_bias_v']


def load_source_config():
    return yaml.safe_load((ROOT / 'config/source_side.yaml').read_text(encoding='utf-8'))


@dataclass
class SourceDeviceParams:
    isolation_db: float
    damage_threshold_nw: float
    temperature_c: float
    intensity_scale: float
    susceptibility: float


@dataclass
class SourceState:
    photo: float = 0.0
    damage_db: float = 0.0
    seed_response: float = 0.0
    bias: float = 0.0


@dataclass
class SensorParams:
    unit: str
    noise: float
    low: float
    high: float
    period_s: float
    response_s: float


@dataclass
class AttackSchedule:
    scenario: int
    start_s: float
    stop_s: float
    second_s: float
    power_nw: float
    wavelength_nm: float
    intermittent: bool = False
    ramp_s: float = 0.0
    repair_s: float | None = None
    second_power_ratio: float = 0.03
    second_wavelength_nm: float | None = None
    intermittent_period_s: float = 20.

    def validate(self, scenario):
        values = [self.start_s, self.stop_s, self.second_s, self.power_nw,
                  self.wavelength_nm, self.ramp_s, self.second_power_ratio]
        if not np.isfinite(values).all() or self.scenario != scenario:
            raise ValueError('invalid schedule or scenario mismatch')
        if not 0 <= self.start_s < self.stop_s or self.power_nw < 0 or self.wavelength_nm <= 0 or self.ramp_s < 0:
            raise ValueError('invalid schedule bounds')
        if scenario == 5 and not self.start_s < self.second_s < self.stop_s:
            raise ValueError('chain transition must lie inside exposure')
        if self.repair_s is not None and (not np.isfinite(self.repair_s) or self.repair_s < 0):
            raise ValueError('invalid repair time')
        if self.second_power_ratio < 0 or (self.second_wavelength_nm is not None and
                (not np.isfinite(self.second_wavelength_nm) or self.second_wavelength_nm <= 0)):
            raise ValueError('invalid chain second stage')
        if not np.isfinite(self.intermittent_period_s) or self.intermittent_period_s<=0:
            raise ValueError('invalid intermittent period')

    def input_at(self, t):
        if self.scenario == 0 or not self.start_s <= t < self.stop_s:
            return 0.0, 0
        mechanism = self.scenario
        if mechanism == 5:
            mechanism = 4 if t < self.second_s else 1
        duty = 0.0 if self.intermittent and int((t-self.start_s)/self.intermittent_period_s) % 2 else 1.0
        ramp = min(1.0, (t-self.start_s)/self.ramp_s) if self.ramp_s > 0 else 1.0
        power = self.power_nw * (self.second_power_ratio if self.scenario == 5 and mechanism == 1 else 1.0)
        return power * duty * ramp, mechanism


@dataclass
class SourceSession:
    observations: pd.DataFrame
    truth: pd.DataFrame
    metadata: dict = field(default_factory=dict)


class SourceSideSimulator:
    def __init__(self, config=None):
        self.config = copy.deepcopy(config or load_source_config())
        c = self.config
        if not np.isfinite(c['dt_s']) or c['dt_s'] <= 0 or int(c['samples']) != c['samples'] or c['samples'] < 4:
            raise ValueError('invalid sample count or cadence')
        if not 0 <= c['missing_probability'] <= 1:
            raise ValueError('invalid missing probability')
        for key, values in c['sensors'].items():
            _, noise, low, high, period, response = values
            if not np.isfinite(values[1:]).all() or noise < 0 or low >= high or period < c['dt_s'] or response <= 0:
                raise ValueError('invalid sensor configuration: ' + key)
        if any(w < 2*c['dt_s'] for w in c['windows_s']):
            raise ValueError('windows require at least two cadence samples')
        for key in ('photo_recovery_s','seeding_response_s','seeding_scale_nw','controller_response_s','nominal_mu'):
            if not np.isfinite(c['source'][key]) or c['source'][key]<=0:
                raise ValueError('invalid positive source parameter: '+key)
        if round(c['experiment']['receiver_effective_pulses_per_s']*c['dt_s']) < 1:
            raise ValueError('receiver integration must contain at least one pulse')

    def simulate(self, seed=42, device_id=0, session_id=0, scenario=0,
                 profile='minimal', schedule=None, stress='nominal'):
        if scenario not in range(6) or profile not in ('minimal', 'instrumented'):
            raise ValueError('unknown scenario or sensor profile')
        if stress not in ('nominal','startup','weak','spectral_blind','intermittent','slow',
                          'sensor_failure','offset','environment'):
            raise ValueError('unknown stress')
        c = self.config; p = c['source']; a = c['assumptions']; n = int(c['samples']); dt = float(c['dt_s'])
        if n < 2 or dt <= 0:
            raise ValueError('samples >= 2 and dt_s > 0 required')
        # Device draws do not depend on scenario/session. Other streams do not depend
        # on scenario either: counterfactual sessions share environment and noise.
        dr = np.random.default_rng(np.random.SeedSequence([seed, device_id, 101]))
        streams = np.random.SeedSequence([seed, device_id, session_id, 202]).spawn(3)
        env, atk, sensor = [np.random.default_rng(s) for s in streams]
        dev = SourceDeviceParams(dr.uniform(*p['isolation_db']), dr.uniform(*p['damage_threshold_nw']),
                                dr.normal(*a['device_temperature']), dr.uniform(*a['device_intensity_scale']), dr.uniform(*a['device_susceptibility']))
        start = float(atk.uniform(*a['onset_fraction'])*n*dt)
        if stress == 'startup': start = 0.0
        if schedule is None:
            # Power/wavelength are attack mechanism inputs, not feature signatures.
            lo, hi = a['attack_power_ranges'][scenario]
            power = 10**atk.uniform(np.log10(lo), np.log10(hi))
            if stress == 'weak': power *= a['weak_multiplier']
            wavelength = float(atk.uniform(*a['attack_wavelength_nm']))
            if scenario == 3: wavelength = float(atk.uniform(*a['photo_wavelength_nm']))
            if stress == 'spectral_blind': wavelength = a['blind_wavelength_nm']
            schedule = AttackSchedule(scenario, start, n*dt*a['stop_fraction'], n*dt*a['second_fraction'], power,
                                      wavelength, stress == 'intermittent',
                                      n*dt*a['slow_ramp_fraction'] if stress == 'slow' else 0,
                                      second_power_ratio=c['experiment']['chain_second_power_ratio'],
                                      second_wavelength_nm=c['experiment']['chain_second_wavelength_nm'],
                                      intermittent_period_s=a['intermittent_period_s'])
        schedule.validate(scenario)
        keys = MINIMAL if profile == 'minimal' else list(c['sensors'])
        sensors = {k: SensorParams(*c['sensors'][k]) for k in keys}
        # Pre-draw noise so attack-dependent state never shifts the RNG stream.
        # Stable identity, independent of sensor list order, profile and process hash seed.
        noise, missing, offsets, gains = {}, {}, {}, {}
        for k in keys:
            identity = int.from_bytes(hashlib.sha256(k.encode()).digest()[:4], 'little')
            sensor = np.random.default_rng(np.random.SeedSequence([seed,device_id,session_id,404,identity]))
            noise[k] = sensor.normal(size=n)
            missing[k] = sensor.random(n) < (c['experiment']['failure_probability'] if stress == 'sensor_failure' else c['missing_probability'])
            offsets[k] = float(sensor.normal()) * (a['offset_multiplier'] if stress == 'offset' else 1)
            gains[k] = float(sensor.normal(1, c['experiment']['sensor_gain_sd']))
        protocol_rng=np.random.default_rng(np.random.SeedSequence([seed,device_id,session_id,303]))
        protocol_uniforms=protocol_rng.random((n,2))
        thermal = np.zeros(n)
        for i in range(1,n): thermal[i] = a['thermal_ar']*thermal[i-1] + env.normal(0,a['thermal_innovation'])
        if stress == 'environment': thermal *= a['environment_multiplier']
        change = int(env.integers(n//4, 3*n//4))
        level = env.uniform(*a['intensity_step'])
        maintenance = int(env.integers(n//3, 2*n//3))
        optical = env.lognormal(*a['optical_lognormal'], n)
        optical[maintenance:maintenance+a['optical_burst_samples']] += env.uniform(*a['optical_burst'])
        # Sensors start settled at a nominal, unexposed device, including startup attacks.
        filtered = dict(watchdog_power=optical[0], emitted_intensity=p['nominal_mu']*dev.intensity_scale,
            temperature=dev.temperature_c, modulator_bias_v=0., isolation_db=dev.isolation_db,
            extinction_ratio_db=a['extinction_base'], smsr_db=a['smsr_base'], rin=a['rin_base'], wavelength_shift_pm=0.)
        state = SourceState(); rows=[]; truth=[]; held={}; last={}; clipped={}
        for i in range(n):
            t = i*dt
            power, mechanism = schedule.input_at(t)
            wavelength = (schedule.second_wavelength_nm if scenario == 5 and t >= schedule.second_s
                          and schedule.second_wavelength_nm is not None else schedule.wavelength_nm)
            if schedule.repair_s is not None and t >= schedule.repair_s and t-dt < schedule.repair_s:
                state.damage_db=0.
            # Isolation spectral response is an explicit assumed vulnerability.
            spectral_loss = min(a['spectral_loss_cap_db'], abs(wavelength-a['spectral_center_nm'])*a['spectral_db_per_nm'])
            isolation = max(0., dev.isolation_db-state.damage_db-spectral_loss)
            internal = power*10**(-isolation/10)
            damage_drive = max(0., power/dev.damage_threshold_nw-1)
            state.damage_db = min(a['damage_cap_db'], state.damage_db + dt*p['damage_rate_db_s']*damage_drive*dev.susceptibility)
            photo_drive = internal*np.exp(-((wavelength-a['photo_center_nm'])/a['photo_width_nm'])**2)
            rate = p['photo_growth_per_nw_s']*photo_drive*dev.susceptibility
            decay = 1/p['photo_recovery_s']
            equilibrium = rate/(rate+decay)
            state.photo = equilibrium+(state.photo-equilibrium)*np.exp(-(rate+decay)*dt)
            seed_drive = internal/p['seeding_scale_nw']*np.exp(-((wavelength-a['spectral_center_nm'])/a['seeding_width_nm'])**2)
            target = min(seed_drive, a['seeding_cap'])
            state.seed_response += (target-state.seed_response)*(1-np.exp(-dt/p['seeding_response_s']))
            temp = dev.temperature_c+thermal[i]+a['warmup_temperature']*np.exp(-t/a['warmup_tau_s'])
            phase_shift = a['phase_temperature']*(temp-a['device_temperature'][0])+a['phase_photo']*state.photo
            state.bias += (phase_shift-state.bias)*(1-np.exp(-dt/p['controller_response_s']))
            maintenance_bias = a['maintenance_bias']*np.exp(-max(0,t-maintenance*dt)/a['maintenance_tau_s']) if i>=maintenance else 0
            bias = state.bias+maintenance_bias
            error = phase_shift-bias
            nominal = p['nominal_mu']*(1+(level if i>=change else 0))
            actual_mu = nominal*dev.intensity_scale*(1+a['intensity_temperature']*(temp-a['device_temperature'][0])+a['intensity_seeding']*state.seed_response)
            # Receiver observable adapter: intensity + modulation residual affect gain/QBER.
            det_gain,qber=bb84_source_observables(actual_mu,error,eta=a['receiver_eta'],dark=a['receiver_dark'],base_error=a['receiver_misalignment'])
            watchdog_response = np.exp(-((wavelength-a['spectral_center_nm'])/p['sensor_spectral_width_nm'])**2)
            physical = dict(watchdog_power=optical[i]+power*p['watchdog_tap']*watchdog_response,
                            emitted_intensity=actual_mu, temperature=temp, modulator_bias_v=bias,
                            isolation_db=dev.isolation_db-state.damage_db-a['isolation_drift_db_s']*t,
                            extinction_ratio_db=a['extinction_base']-a['extinction_error']*abs(error)-a['extinction_photo']*state.photo,
                            smsr_db=a['smsr_base']-a['smsr_seeding']*state.seed_response+a['smsr_temperature']*(temp-a['device_temperature'][0]),
                            rin=a['rin_base']+a['rin_seeding']*state.seed_response+a['rin_temperature']*abs(temp-a['device_temperature'][0]),
                            wavelength_shift_pm=a['wavelength_temperature']*(temp-a['device_temperature'][0])+a['wavelength_seeding']*state.seed_response)
            row={'time_s': t, 'nominal_intensity':nominal}
            # Measurement noise at receiver, no true component state exposed.
            from scipy.stats import binom
            pulses = int(round(c['experiment']['receiver_effective_pulses_per_s']*dt))
            received=int(binom.ppf(protocol_uniforms[i,0],pulses,det_gain))
            errors=int(binom.ppf(protocol_uniforms[i,1],received,qber)) if received else 0
            row['protocol_gain']=received/pulses
            row['protocol_qber']=errors/received if received else np.nan
            row['protocol_pulses']=pulses
            row['protocol_detections']=received
            for j,(k,s) in enumerate(sensors.items()):
                filtered[k] = filtered.get(k,physical[k])+(physical[k]-filtered.get(k,physical[k]))*(1-np.exp(-dt/s.response_s))
                due = i == 0 or int(t/s.period_s) != int((t-dt)/s.period_s)
                if due and not missing[k][i]:
                    value = filtered[k]*gains[k]+s.noise*(noise[k][i]+offsets[k])
                    clipped[k] = value < s.low or value > s.high
                    held[k] = float(np.clip(value,s.low,s.high)); last[k]=t
                row[k] = held.get(k,np.nan)
                row[k+'__age_s'] = t-last[k] if k in last else t+dt
                row[k+'__missing'] = int(due and missing[k][i])
                row[k+'__saturated'] = int(clipped.get(k,False))
                row[k+'__stale'] = int(k not in last or t-last[k] >= s.period_s*c['experiment']['stale_periods'])
            rows.append(row)
            # Current label is physical mechanism/effect, never future scenario.
            label = mechanism if power > 0 else (4 if state.damage_db>a['damage_label_threshold'] else
                    3 if state.photo>a['photo_label_threshold'] else 2 if state.seed_response>a['seed_label_threshold'] else 0)
            truth.append(dict(time_s=t, family=label, active=int(power>0),
                              requested_mechanism=mechanism, incident_power_nw=power, wavelength_nm=wavelength,
                              damage_effect=int(state.damage_db>a['damage_label_threshold']), photo_effect=int(state.photo>a['photo_label_threshold']),
                              seeding_effect=int(state.seed_response>a['seed_label_threshold']), seed_state=state.seed_response,
                              degraded=int(state.damage_db>a['damage_label_threshold'] or state.photo>a['photo_label_threshold'] or state.seed_response>a['seed_label_threshold']),
                              damage_db=state.damage_db, photo_state=state.photo,
                              internal_power_nw=internal, returned_power_nw=internal*p['reflection']*10**(-isolation/10),
                              actual_mu=actual_mu, nominal_mu=nominal, protocol_qber=qber,
                              protocol_gain=det_gain))
        return SourceSession(pd.DataFrame(rows),pd.DataFrame(truth),
            dict(seed=seed,device_id=device_id,session_id=session_id,scenario=scenario,profile=profile,
                 stress=stress,dt_s=dt,start_s=schedule.start_s,stop_s=schedule.stop_s,
                 second_s=schedule.second_s,provenance='SIMULATED DATA',
                 schedule=asdict(schedule),device=asdict(dev),sensors={k:asdict(v) for k,v in sensors.items()},
                 sensor_offsets=offsets,sensor_gains=gains,config=copy.deepcopy(c),
                 rng='SeedSequence device/environment/attack/sensor streams',
                 assumptions='config/source_side.yaml; docs/PHASE3.md'))
