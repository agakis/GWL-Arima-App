import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent / "app.py"

    # Start Streamlit app
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    subprocess.Popen(cmd)

    # Give Streamlit a moment to start
    time.sleep(3)

    # Open browser
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    main()