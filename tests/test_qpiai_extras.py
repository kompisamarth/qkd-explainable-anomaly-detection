"""The optional QpiAI module must work without an account and never leak a key."""

import numpy as np

from qpiai_extras import backend
from qpiai_extras.bb84_circuit import run_bb84_rounds
from qpiai_extras.qrng import QRNGSource


def test_key_lookup_order_and_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("QPIAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    env = tmp_path / "qcloud.env"
    monkeypatch.setattr(backend, "ENV_FILE", env)
    assert backend.find_api_key() == (None, "none")
    env.write_text('API_KEY="file-key"\n')
    assert backend.find_api_key() == ("file-key", "qcloud.env")
    monkeypatch.setenv("QPIAI_API_KEY", "env-key")
    assert backend.find_api_key() == ("env-key", "environment variable QPIAI_API_KEY")


def test_status_never_contains_the_key(monkeypatch):
    monkeypatch.setenv("QPIAI_API_KEY", "secret-123")
    st = backend.qpiai_status(login=False)
    assert "secret-123" not in repr(st)


def test_qrng_falls_back_to_numpy_without_an_account(monkeypatch):
    monkeypatch.delenv("QPIAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(backend, "ENV_FILE", backend.ROOT_DIR / "does-not-exist.env")
    q = QRNGSource(fallback_seed=1)
    bits = q.integers(0, 2, size=(4, 1000), dtype=np.int8)
    assert bits.shape == (4, 1000) and set(np.unique(bits)) <= {0, 1}
    assert abs(bits.mean() - 0.5) < 0.03
    if not q.status.usable:
        assert "fallback" in q.source_label


def test_circuit_fallback_reproduces_bb84_physics():
    clean = run_bb84_rounds(600, p_eve=0.0, seed=2, force_fallback=True)
    full = run_bb84_rounds(2000, p_eve=1.0, seed=2, force_fallback=True)
    assert clean["qber"] == 0.0
    assert full["within_3_sigma"], full
