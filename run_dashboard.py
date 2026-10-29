"""Launch the QKD Security Operations Center dashboard.

    python run_dashboard.py              # http://127.0.0.1:8000
    python run_dashboard.py --port 9000 --reload

Replaces the previous Streamlit entry point (app.py). The UI is a static
frontend in frontend/ served by the FastAPI app in backend/server.py.
"""

import argparse
import sys
import webbrowser
from pathlib import Path
from threading import Timer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-reload on source changes")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    ckpt = ROOT / "data" / "processed" / "checkpoints" / "xgb_multiattack.joblib"
    if not ckpt.exists():
        print("WARNING: model checkpoint missing. Run scripts 06 and 07 first.", file=sys.stderr)
    if not (ROOT / "data" / "processed" / "trial_sessions").exists():
        print("WARNING: trial sessions missing. Run scripts/11_mdi_protocol_comparison.py.", file=sys.stderr)

    url = f"http://{args.host}:{args.port}"
    print("=" * 62)
    print("  QKD Security Operations Center")
    print(f"  {url}")
    print("=" * 62)
    if not args.no_browser:
        Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run("backend.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
