import subprocess
from pathlib import Path

SARAD_ROOT = Path(__file__).resolve().parent / "external"

def train(dataset="SMD", device="gpu"):
    cmd = [
        "python",
        "src/train.py",
        f"data={dataset}",
        f"trainer={device}",
    ]
    print ("Here")
    subprocess.run(cmd, check=True, cwd=SARAD_ROOT)