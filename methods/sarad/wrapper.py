import subprocess
from pathlib import Path

SARAD_ROOT = Path(__file__).resolve().parent / "external"

def train(use_processed, processed_path, dataset="SMD", device="gpu"):
    cmd = [
        "python",
        "src/train.py",
        f"data.use_processed={use_processed}",
        f"data.processed_path={processed_path}",
        f"data={dataset}",
        f"trainer={device}",
        "hydra.run.dir=outputs/sarad_run",
    ]
    print ("Here")
    subprocess.run(cmd, check=True, cwd=SARAD_ROOT)