"""Dashboard SHAP tab: per-feature context and per-block threshold verdicts."""

import pytest

from src.features.time_series import FEATURE_CONTEXT_63, FEATURE_NAMES_63


def test_every_feature_has_a_short_context():
    assert set(FEATURE_CONTEXT_63) == set(FEATURE_NAMES_63)
    for name, text in FEATURE_CONTEXT_63.items():
        assert 4 <= len(text.split()) <= 5, f"{name}: '{text}' should be 4-5 words"


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from backend import server
    if server._model is None:
        pytest.skip("model checkpoint not generated (run scripts 06 and 07)")
    return TestClient(server.app)


def _explain(client, cls, block_pos=-1):
    s = client.get(f"/api/trial/{cls}")
    if s.status_code == 404:
        pytest.skip("trial sessions not generated")
    s = s.json()
    d = client.get("/api/explain", params={"session_id": s["session_id"], "block": s["blocks"][block_pos]})
    assert d.status_code == 200
    return s, d.json()


def test_explain_returns_all_63_attribution_vectors(client):
    _, d = _explain(client, 1)
    feats = d["features"]
    assert sorted(f["feature"] for f in feats) == sorted(FEATURE_NAMES_63)
    assert [f["rank"] for f in feats] == list(range(1, 64))
    mags = [abs(f["shap"]) for f in feats]
    assert mags == sorted(mags, reverse=True)
    assert all(f["context"] == FEATURE_CONTEXT_63[f["feature"]] for f in feats)
    assert d["top_features"] == feats[:14]


def test_threshold_verdicts_are_consistent(client):
    s, d = _explain(client, 3)
    checks = {t["name"]: t for t in d["thresholds"]}
    # 11% (mixed, Z, X), Hoeffding, finite-size Wilson (calibrated sessions carry a CI), XGBoost
    assert len(checks) == 6
    assert "Finite-size 11% (Wilson)" in checks
    for t in d["thresholds"]:
        assert abs(t["margin"] - (t["value"] - t["threshold"])) < 1e-12
        if t["unit"] == "pct":
            assert t["exceeded"] == (t["value"] > t["threshold"])
    classical = checks["Classical 11% abort"]
    assert classical["threshold"] == 0.11
    ml = next(t for t in d["thresholds"] if t["name"].startswith("XGBoost"))
    assert ml["exceeded"] == (d["pred"] != 0)
    assert abs(ml["value"] - (1 - d["probs"][0])) < 1e-9
