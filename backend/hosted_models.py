"""Validate the inference-only subset of a completed Phase 3 run."""

from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import platform

from src.data.source_artifacts import PACKAGES, SCHEMA, digest


def source_digest(path, normalize_newlines=False):
    content = path.read_bytes()
    if normalize_newlines:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def validate_bundle(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA or manifest["status"] != "inference":
        raise ValueError("Incompatible hosted model bundle")
    expected = manifest["versions"]
    if platform.python_version_tuple()[:2] != tuple(expected["python"].split(".")[:2]):
        raise ValueError("Hosted model requires the recorded Python minor version")
    for name in PACKAGES:
        if name == "pyarrow":
            continue  # Replay-only dependency.
        try:
            actual = version(name)
        except PackageNotFoundError as exc:
            if name != "xgboost":
                raise ValueError(f"Missing hosted model dependency: {name}") from exc
            try:
                actual = version("xgboost-cpu")
            except PackageNotFoundError:
                raise ValueError("Missing hosted model dependency: xgboost") from exc
        if actual != expected[name]:
            raise ValueError(f"Incompatible hosted model dependency: {name}")
    for name, expected_hash in manifest["checksums"].items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file() or digest(target) != expected_hash:
            raise ValueError(f"Corrupt hosted artifact: {name}")
    for name, expected_hash in manifest["source_hashes"].items():
        if source_digest(directory.parent / name, manifest.get("normalize_source_newlines", False)) != expected_hash:
            raise ValueError(f"Stale hosted artifact: {name}")
    return manifest
