"""Compare one saved clean prediction against the unmodified official test.py."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiment import ROOT, UPSTREAM, read_json, write_json


def verify(output):
    output = Path(output).resolve()
    metadata = read_json(output / "metadata.json")
    manifest = read_json(output / "manifest.json")
    sample_id = metadata["ids"][0]
    row = next(r for r in manifest["rows"] if r["id"] == sample_id)
    destination = output / "upstream_reference.npz"
    if destination.exists():
        destination.unlink()  # only this script's generated reference; rerun upstream each time
    command = [sys.executable, "test.py", "-g", "0", "-in", str(Path(manifest["data_root"]) / row["image"]),
               "-out", str(destination), "-exp", "trufor_ph3", "--save_np",
               "TEST.MODEL_FILE", metadata["checkpoint"]]
    subprocess.run(command, cwd=UPSTREAM, check=True)
    result = {"id": sample_id, "command": command, "rtol": 1e-5, "atol": 1e-6, "max_absolute_errors": {}}
    with np.load(destination, allow_pickle=False) as official, np.load(output / "raw/clean" / (sample_id + ".npz"), allow_pickle=False) as wrapper:
        for key in ("map", "conf", "score", "np++"):
            np.testing.assert_allclose(wrapper[key], official[key], rtol=1e-5, atol=1e-6)
            result["max_absolute_errors"][key] = float(np.max(np.abs(wrapper[key] - official[key])))
    result["passed"] = True
    write_json(output / "upstream_equivalence.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
