"""Locate the QpiAI SDK and an API key without ever requiring them."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("qpiai_extras")
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / "qcloud.env"


@dataclass
class QpiAIStatus:
    sdk_installed: bool
    api_key_found: bool
    key_source: str
    usable: bool
    reason: str


def _read_env_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("API_KEY") and "=" in line:
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def find_api_key() -> tuple:
    """Returns (key or None, source description). The key itself is never logged."""
    for var in ("QPIAI_API_KEY", "API_KEY"):
        if os.environ.get(var, "").strip():
            return os.environ[var].strip(), f"environment variable {var}"
    key = _read_env_file(ENV_FILE)
    if key:
        return key, "qcloud.env"
    return None, "none"


def qpiai_status(login: bool = True) -> QpiAIStatus:
    """Check SDK + key; with login=True, authenticate so circuits can run."""
    try:
        import qpiai_quantum  # noqa: F401
    except ImportError:
        st = QpiAIStatus(False, False, "none", False,
                         "qpiai-quantum is not installed (pip install -r qpiai_extras/requirements.txt)")
        log.info("QpiAI unavailable: %s; using the NumPy/Qiskit fallback.", st.reason)
        return st
    key, source = find_api_key()
    if key is None:
        st = QpiAIStatus(True, False, "none", False,
                         "no API key (set QPIAI_API_KEY or create qcloud.env with API_KEY=...)")
        log.info("QpiAI unavailable: %s; using the NumPy/Qiskit fallback.", st.reason)
        return st
    if login:
        try:
            from qpiai_quantum import QpiAIQuantumAuth
            QpiAIQuantumAuth.login(api_key=key)
        except Exception as exc:  # network errors, invalid key, ...
            st = QpiAIStatus(True, True, source, False, f"login failed: {type(exc).__name__}")
            log.warning("QpiAI login failed (%s); using the NumPy/Qiskit fallback.", type(exc).__name__)
            return st
    log.info("QpiAI SDK ready (key from %s).", source)
    return QpiAIStatus(True, True, source, True, "ok")
