import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent

subprocess.run(
    [sys.executable, "-m", "streamlit", "run",
     str(repo_root / "app.py"), "--server.headless", "false"],
    cwd=str(repo_root),
)
