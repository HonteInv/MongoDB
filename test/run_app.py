"""
run_app.py — launch either the query app or admin app via Streamlit.

Usage (from the project root):
    python test/run_app.py          # defaults to query app
    python test/run_app.py query    # Portfolio AI (query_app.py)
    python test/run_app.py admin    # Admin upload app (admin_app.py)
"""

import subprocess
import sys
import os

# ── Resolve project root (one level up from this file) ─────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APPS = {
    "query": {
        "file":        os.path.join(ROOT, "query_app.py"),
        "label":       "Portfolio AI  (query_app.py)",
        "port":        8501,
    },
    "admin": {
        "file":        os.path.join(ROOT, "admin_app.py"),
        "label":       "Admin Upload  (admin_app.py)",
        "port":        8502,
    },
}

def main():
    # Parse argument — default to "query"
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "query"

    if arg not in APPS:
        print(f"Unknown app '{arg}'. Choose: {', '.join(APPS)}")
        sys.exit(1)

    app  = APPS[arg]
    port = app["port"]

    print(f"\n  Starting: {app['label']}")
    print(f"  URL:      http://localhost:{port}")
    print(f"  File:     {app['file']}")
    print(f"\n  Press Ctrl+C to stop.\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        app["file"],
        "--server.port", str(port),
        "--server.headless", "false",
    ])

if __name__ == "__main__":
    main()
