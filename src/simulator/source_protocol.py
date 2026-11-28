"""Limited analytic bridge to existing optical click models; NOT a security bound."""
import numpy as np
from src.simulator.calibrated import coherent_clicks
from src.simulator.mdi_qkd import MDIQKD


def bb84_source_observables(actual_mu, modulation_error, eta=.02, dark=1e-7, base_error=.015):
    ed=base_error+.25*np.sin(modulation_error)**2
    gain,errors=coherent_clicks(actual_mu*eta*(1-ed),actual_mu*eta*ed,dark,dark)
    return float(gain),float(errors/max(gain,1e-15))


def mdi_source_observables(alice_mu,bob_mu,distance_km=50):
    """Actual source intensities at both endpoints; only gain/error, no key bound."""
    model=MDIQKD(); eta=model.ch.arm_transmittance(distance_km)
    gain,error=model.wcp_stats(alice_mu,bob_mu,eta,eta,0)
    return dict(gain=gain,qber=error,label='ANALYTIC MODEL',
                limitation='Intensity-only adapter; no source-leakage security bound.')
