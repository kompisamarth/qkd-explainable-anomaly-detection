"""Optional QpiAI Quantum SDK integration (qpiai-quantum on PyPI, Apache-2.0).

Nothing in the main pipeline imports this package. Every entry point works without a
QpiAI account: if the SDK is not installed or no API key is found, it logs why and falls
back to NumPy (random bits) or Qiskit / analytic statevectors (circuits).

API key lookup order (the key is never written anywhere by this code):
  1. environment variable QPIAI_API_KEY
  2. environment variable API_KEY (the SDK's own convention)
  3. a qcloud.env file in the repository root containing  API_KEY="..."
qcloud.env is listed in .gitignore. Never commit a key.
"""

from qpiai_extras.backend import QpiAIStatus, qpiai_status
from qpiai_extras.qrng import QRNGSource

__all__ = ["QpiAIStatus", "qpiai_status", "QRNGSource"]
