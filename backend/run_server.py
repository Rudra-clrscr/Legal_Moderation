"""Start the API server with Python UTF-8 mode enabled.

Marker's OCR engine reads UTF-8 files with the locale codec, which breaks on Windows unless UTF-8
mode is on. This launcher re-executes itself with ``-X utf8`` when needed, then runs uvicorn.

    python run_server.py [--host 127.0.0.1] [--port 8000]
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    if not sys.flags.utf8_mode:
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")  # also for any worker processes
    uvicorn.run("app.main:app", host=args.host, port=args.port)
