"""Create a disposable Vercel upload from a trusted, completed Phase 3 run."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.source_artifacts import PACKAGES, digest, validate_run, write_json


def prepare(run, output):
    manifest = validate_run(run)
    if manifest["mode"] == "smoke":
        raise ValueError("Choose a bounded or full run for inference")
    output.mkdir(parents=True, exist_ok=False)

    def copy(source, name):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    for directory in ("backend", "src", "frontend", "config", "results/metrics",
                      "data/processed/trial_sessions"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".html", ".css", ".js", ".yaml", ".json", ".csv"}:
                copy(path, path.relative_to(ROOT))
    copy(ROOT / "config.yaml", "config.yaml")
    for name in manifest["source_hashes"]:
        copy(ROOT / name, name)
    for filename in ("train_multiattack.meta.json", "test_multiattack.meta.json",
                     "heldout_generalization.meta.json", "sessions_multiattack.csv"):
        path = ROOT / "data/processed" / filename
        if path.is_file():
            copy(path, path.relative_to(ROOT))
    checkpoint = Path("data/processed/checkpoints/xgb_multiattack.joblib")
    copy(ROOT / checkpoint, checkpoint)

    inference = {name: manifest[name] for name in (
        "schema", "versions", "deployment", "source_hashes", "provenance")}
    inference.update(status="inference", run_id=run.name, feature_schemas={}, checksums={})
    for filename in [*inference["deployment"].values(), "summary.csv"]:
        destination = Path("hosted_models") / filename
        copy(run / filename, destination)
        inference["checksums"][filename] = digest(output / destination)
        if filename in manifest["feature_schemas"]:
            inference["feature_schemas"][filename] = manifest["feature_schemas"][filename]
    write_json(output / "hosted_models/manifest.json", inference)

    (output / "app.py").write_text("from backend.server import app\n", encoding="utf-8")
    (output / ".python-version").write_text("3.13\n", encoding="utf-8")
    (output / ".vercelignore").write_text(".env*\n.vercel\n.git\n**/__pycache__\n", encoding="utf-8")
    pins = dict(line.split("==", 1) for line in
                (ROOT / "requirements-phase3-lock.txt").read_text().splitlines() if "==" in line)
    dependencies = [p for p in PACKAGES if p not in ("pyarrow", "xgboost")]
    dependencies += ["fastapi", "pydantic", "pyyaml"]
    lines = [f"{package}=={pins[package]}" for package in dependencies]
    lines.extend([
        f"xgboost-cpu=={pins['xgboost']}; sys_platform == 'linux'",
        f"xgboost=={pins['xgboost']}; sys_platform != 'linux'",
    ])
    (output / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "vercel.json", {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "framework": "fastapi",
        "functions": {"app.py": {"maxDuration": 60}},
    })
    print(f"Vercel upload prepared at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed Phase 3 run directory")
    parser.add_argument("--output", type=Path, default=ROOT / ".vercel-qkd")
    args = parser.parse_args()
    prepare(args.run.resolve(), args.output.resolve())
