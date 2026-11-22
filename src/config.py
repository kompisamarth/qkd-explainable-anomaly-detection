"""Configuration loading shared by every script.

`config.yaml` holds pipeline and ML settings; `config/physics.yaml` holds every
physical parameter (with sources). `load_config()` merges them so older scripts
keep reading `cfg["physics"]` and `cfg["tf_qkd"]["channel"]` unchanged.
"""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
PHYSICS_PATH = ROOT_DIR / "config" / "physics.yaml"


def load_physics() -> Dict[str, Any]:
    with open(PHYSICS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    phys = load_physics()
    cfg["physics"] = copy.deepcopy(phys["legacy_bb84"])
    cfg.setdefault("tf_qkd", {})["channel"] = copy.deepcopy(phys["tf_qkd"]["channel"])
    cfg["tf_qkd"]["bb84_misalignment"] = phys["tf_qkd"]["bb84_misalignment"]
    cfg["mdi_qkd"] = copy.deepcopy(phys["mdi_qkd"])
    cfg["calibrated_bb84"] = copy.deepcopy(phys["calibrated_bb84"])
    return cfg


def config_hash() -> str:
    """Short SHA-256 over both config files, recorded in dataset metadata."""
    h = hashlib.sha256()
    for p in (CONFIG_PATH, PHYSICS_PATH):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def git_commit() -> str:
    """Current commit, with '-dirty' when tracked files are modified."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT_DIR,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)
