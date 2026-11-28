"""Fail a Git deployment if a published checkpoint is missing or incompatible."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.hosted_models import validate_bundle
from src.data.source_artifacts import load_model
from src.features.time_series import FEATURE_NAMES_63
import joblib


def main():
    directory = ROOT / "hosted_models"
    manifest = validate_bundle(directory)
    bb84 = manifest["bb84_checkpoint"]
    if bb84 not in manifest["checksums"]:
        raise ValueError("BB84 checkpoint has no checksum")
    if joblib.load(directory / bb84).n_features_in_ != len(FEATURE_NAMES_63):
        raise ValueError("Invalid BB84 feature schema")
    for profile in ("minimal", "instrumented"):
        model = load_model(directory, manifest["deployment"][profile], manifest)
        if model.profile != profile:
            raise ValueError(f"Unexpected source checkpoint profile: {profile}")
    print("Vercel release validated: BB84, minimal, and instrumented checkpoints.")


if __name__ == "__main__":
    main()
