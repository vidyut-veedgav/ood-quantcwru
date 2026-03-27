import subprocess
from pathlib import Path

SARAD_ROOT = Path(__file__).resolve().parent / "external"

def train(data_dir, dataset="SMD", device="gpu"):
    cmd = [
        "python",
        "src/train.py",
        f"data.data_dir={data_dir}",
        f"data={dataset}",
        f"trainer={device}",
    ]
    print ("Here")
    subprocess.run(cmd, check=True, cwd=SARAD_ROOT)