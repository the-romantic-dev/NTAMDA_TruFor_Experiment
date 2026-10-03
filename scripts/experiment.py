"""Reproduce TruFor on CocoGlide. See README.md for the scientific protocol."""
import argparse
import csv
from contextlib import redirect_stderr
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from fetch_assets import ROOT, UPSTREAM_COMMIT, digest

UPSTREAM = ROOT / "third_party/TruFor/TruFor_train_test"
PAPER = {"f1_best": 0.720, "f1_fixed": 0.523, "pixel_auc": None}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    paths = list(UPSTREAM.rglob("*.py")) + list((UPSTREAM / "lib/config").glob("*.yaml"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def prepare(data, output, expected=512):
    data = Path(data).resolve()
    with (data / "table.csv").open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != expected:
        raise ValueError(f"Expected {expected} author-table rows, got {len(source)}")
    rows = []
    for row in source:
        image_path, mask_path = [(data / row[k]).resolve() for k in ("fake", "mask")]
        if any(not p.is_relative_to(data) for p in (image_path, mask_path)):
            raise ValueError("Dataset path escapes data root")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            gt = np.array(mask.convert("L"))
            if not set(np.unique(gt)).issubset({0, 255}) or gt.shape != (image.height, image.width):
                raise ValueError(f"Invalid mask or alignment: {mask_path}")
            if not 0 < np.count_nonzero(gt) < gt.size:
                raise ValueError(f"Forged-only localization requires both classes: {mask_path}")
            rows.append({"id": image_path.stem, "image": row["fake"], "mask": row["mask"],
                         "image_sha256": digest(image_path), "mask_sha256": digest(mask_path),
                         "height": image.height, "width": image.width,
                         "forged_pixels": int(np.count_nonzero(gt)), "prompt": row.get("prompt", "")})
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    manifest = {"dataset": "CocoGlide", "version": "author ZIP", "data_root": str(data),
                "table_sha256": digest(data / "table.csv"), "count": len(rows),
                "mask_policy": "white 255=forged; black 0=pristine",
                "rows": sorted(rows, key=lambda r: r["id"])}
    if (data / "download.json").exists():
        manifest["download"] = read_json(data / "download.json")
    write_json(output, manifest)
    return manifest


def official_metrics():
    spec = importlib.util.spec_from_file_location("trufor_official_metrics", UPSTREAM / "metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(prediction, gt):
    prediction, gt = np.asarray(prediction), np.asarray(gt)
    if not np.isin(gt, [0, 1]).all():
        raise ValueError("Ground truth must be binary 0/1")
    gt = gt.astype(bool)
    if prediction.shape != gt.shape or prediction.ndim != 2:
        raise ValueError("Prediction/mask shape mismatch")
    if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
        raise ValueError("Prediction must contain finite probabilities in [0,1]")
    official = official_metrics()
    gt0, gt1 = official.extractGTs(gt)  # unchanged 15px erosion, 11px dilation
    valid = gt0 | gt1
    diagnostics = io.StringIO()
    with redirect_stderr(diagnostics):
        best, fixed = official.computeLocalizationMetrics(prediction, gt)
    # These extra diagnostics use the original orientation. They are not paper F1.
    binary = prediction > 0.5
    tp = int(np.count_nonzero(binary & gt1))
    fp = int(np.count_nonzero(binary & gt0))
    fn = int(np.count_nonzero(~binary & gt1))
    boundary = ~valid
    return {"f1_best": float(best) if np.isfinite(best) else None,
            "f1_fixed": float(fixed) if np.isfinite(fixed) else None,
            "pixel_auc": float(roc_auc_score(gt[valid], prediction[valid])) if gt0.any() and gt1.any() else None,
            "f1_direct": 2 * tp / max(2 * tp + fp + fn, 1),
            "false_positive_rate": fp / int(gt0.sum()) if gt0.any() else None,
            "miss_rate": fn / int(gt1.sum()) if gt1.any() else None,
            "boundary_error": float((binary[boundary] != gt[boundary]).mean()) if boundary.any() else 0.0,
            "valid_pixels": int(valid.sum()),
            "metric_notes": diagnostics.getvalue().strip() + ("; pixel AUC undefined: retained GT has one class" if not (gt0.any() and gt1.any()) else "")}


def load_model(checkpoint, device):
    import torch
    sys.path.insert(0, str(UPSTREAM))
    from lib.config import config
    from lib.models.cmx.builder_np_conf import EncoderDecoder
    cfg = config.clone()
    cfg.defrost()
    cfg.merge_from_file(str(UPSTREAM / "lib/config/trufor_ph3.yaml"))
    cfg.freeze()
    model = EncoderDecoder(cfg=cfg)
    # The legacy checkpoint contains NumPy scalars. Permit pickle only for these exact
    # author-hosted bytes (archive MD5 independently verified during acquisition).
    if digest(checkpoint) != "ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844":
        raise ValueError("Not the verified official checkpoint; refusing legacy pickle loading")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(weights["state_dict"], strict=True)
    return model.to(device).eval(), cfg


def run(manifest_path, output, checkpoint, limit=0, device="cuda:0", seed=42):
    import torch
    manifest_path, output, checkpoint = Path(manifest_path), Path(output), Path(checkpoint)
    manifest = read_json(manifest_path)
    rows = manifest["rows"][:limit] if limit else manifest["rows"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty dataset or duplicate IDs")
    upstream_root = UPSTREAM.parent.resolve()
    git = ["git", "-c", f"safe.directory={upstream_root.as_posix()}", "-C", str(upstream_root)]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT or subprocess.check_output(git + ["status", "--porcelain"], text=True).strip():
        raise ValueError("Use the pinned, unmodified official repository")
    if limit < 0:
        raise ValueError("Limit must be nonnegative")
    data = Path(manifest["data_root"])
    for row in rows:
        for kind in ("image", "mask"):
            if digest(data / row[kind]) != row[kind + "_sha256"]:
                raise ValueError(f"Dataset changed after preparation: {row[kind]}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.enabled = False  # official trufor_ph3.yaml
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    provenance = {"dataset": manifest["dataset"], "manifest_sha256": digest(manifest_path),
                  "checkpoint_sha256": digest(checkpoint), "upstream_commit": UPSTREAM_COMMIT,
                  "upstream_files": source_hashes(), "runner_sha256": digest(__file__),
                  "conditions": ["clean"], "ids": [r["id"] for r in rows], "seed": seed,
                  "torch": torch.__version__, "python": platform.python_version(),
                  "os": platform.platform(), "packages": packages,
                  "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                  "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU",
                  "device": device,
                  "protocol": "official /256; FP32; native resolution; cuDNN off; no AMP/TF32"}
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        previous = read_json(metadata_path)
        if any(previous.get(k) != v for k, v in provenance.items()):
            raise ValueError("Run provenance changed. Choose a new output directory; do not mix predictions.")
    else:
        if any(output.iterdir()):
            raise ValueError("Output is nonempty without metadata; choose a new directory")
        write_json(metadata_path, {**provenance, "command": sys.argv,
                                 "created_utc": datetime.now(timezone.utc).isoformat(),
                                 "checkpoint": str(checkpoint.resolve())})
        write_json(output / "manifest.json", manifest)
        (output / "requirements.freeze.txt").write_text("\n".join(packages), encoding="utf-8")
        (output / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    model, cfg = load_model(checkpoint, device)
    (output / "model_config.yaml").write_text(cfg.dump(), encoding="utf-8")
    started = time.perf_counter()
    failures = []
    total = len(rows)
    condition = "clean"
    for index, row in enumerate(rows):
        path = output / "raw" / condition / (row["id"] + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                if str(existing["image_sha256"]) != row["image_sha256"] or str(existing["mask_sha256"]) != row["mask_sha256"]:
                    raise ValueError(f"Stale prediction: {path}")
                if not all(k in existing for k in ("map", "conf", "np++", "gt", "rgb", "score")):
                    raise ValueError(f"Incomplete prediction: {path}")
            continue
        try:
            with Image.open(data / row["image"]) as image, Image.open(data / row["mask"]) as mask:
                image, gt = image.convert("RGB"), np.array(mask.convert("L")) > 0
            rgb = np.array(image)
            tensor = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float32)[None].to(device) / 256.0
            item_started = time.perf_counter()
            with torch.no_grad():
                pred, conf, det, noise = model(tensor, save_np=True)
                prediction = torch.softmax(pred[0], dim=0)[1].cpu().numpy()
                confidence = torch.sigmoid(conf[0, 0]).cpu().numpy()
                npp = noise[0, 0].cpu().numpy()
                score = torch.sigmoid(det).item()
            if prediction.shape != gt.shape or not np.isfinite(prediction).all():
                raise ValueError("Invalid model output")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, map=prediction, conf=confidence, score=score,
                                    **{"np++": npp}, imgsize=gt.shape, gt=gt, rgb=rgb,
                                    image_sha256=row["image_sha256"], mask_sha256=row["mask_sha256"],
                                    seconds=time.perf_counter() - item_started)
            temporary.replace(path)
            del tensor, pred, conf, det, noise
        except Exception as error:
            failures.append({"id": row["id"], "condition": condition, "error": repr(error)})
            write_json(output / "failures.json", failures)
            print(f"FAILED {condition}/{row['id']}: {error}", flush=True)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        if (index + 1) % 10 == 0 or index == total - 1:
            print(f"{index + 1}/{total}, elapsed {time.perf_counter() - started:.1f}s", flush=True)
    write_json(output / "failures.json", failures)
    write_json(output / "run_status.json", {"complete": not failures, "requested": total,
                                           "failures": len(failures), "seconds_this_invocation": time.perf_counter() - started})
    if failures:
        raise RuntimeError(f"{len(failures)} inference failures; inspect failures.json and rerun to retry")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output):
    output = Path(output)
    metadata = read_json(output / "metadata.json")
    if metadata.get("upstream_files") != source_hashes():
        raise ValueError("Upstream source differs from inference provenance")
    if metadata["conditions"] != ["clean"]:
        raise ValueError("Only the baseline experiment is supported")
    if not metadata["ids"] or len(set(metadata["ids"])) != len(metadata["ids"]):
        raise ValueError("Empty or duplicate evaluation IDs")
    per_image, failures, summaries = [], [], []
    for condition in metadata["conditions"]:
        condition_rows = []
        for sample_id in metadata["ids"]:
            path = output / "raw" / condition / (sample_id + ".npz")
            if not path.exists():
                raise FileNotFoundError(f"Missing prediction: {path}. Resume inference first.")
            with np.load(path, allow_pickle=False) as prediction:
                metrics = measure(prediction["map"], prediction["gt"])
                metrics["mean_confidence"] = float(prediction["conf"].mean())
                metrics["seconds"] = float(prediction["seconds"])
            row = {"condition": condition, "id": sample_id, "status": "ok", **metrics}
            if any(metrics[k] is None for k in PAPER):
                failures.append(row)
            condition_rows.append(row)
        per_image.extend(condition_rows)
        valid = [r for r in condition_rows if r["status"] == "ok"]
        summary = {"condition": condition, "n_requested": len(condition_rows), "n_valid": len(valid),
                   "n_undefined": sum(any(r[k] is None for k in PAPER) for r in valid)}
        for metric in PAPER:
            values = [r[metric] for r in valid if r[metric] is not None]
            summary["n_" + metric] = len(values)
            summary[metric] = float(np.mean(values)) if values else None
        summaries.append(summary)
    keys = list(dict.fromkeys(k for row in per_image for k in row))
    write_csv(output / "metrics/per_image.csv", [{k: r.get(k) for k in keys} for r in per_image])
    write_csv(output / "metrics/summary.csv", summaries)
    write_json(output / "metrics/summary.json", summaries)
    write_json(output / "metrics/undefined.json", failures)
    write_json(output / "metrics/evaluation_provenance.json", {"evaluator_sha256": digest(__file__),
               "official_metrics_sha256": digest(UPSTREAM / "metrics.py"), "numpy": np.__version__,
               "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()})
    comparisons = []
    baseline = next((s for s in summaries if s["condition"] == "clean"), None)
    if baseline:
        for metric, paper in PAPER.items():
            value = baseline[metric]
            comparisons.append({"dataset": metadata["dataset"], "metric": metric, "paper": paper,
                                "reproduction": value, "absolute_delta": abs(value - paper) if value is not None and paper is not None else None,
                                "n_valid": baseline["n_" + metric], "n_requested": baseline["n_requested"],
                                "scope": "full benchmark" if len(metadata["ids"]) == 512 else "subset; not benchmark reproduction"})
        write_csv(output / "metrics/paper_comparison.csv", comparisons)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", type=Path, default=ROOT / "data")
    prep.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    infer = sub.add_parser("run")
    infer.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, default=ROOT / "weights/weights/trufor.pth.tar")
    infer.add_argument("--limit", type=int, default=0, help="0=all; sorted deterministic subset otherwise")
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--seed", type=int, default=42)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.data, args.manifest)
        print(f"Validated {manifest['count']} pairs; manifest: {args.manifest}")
    elif args.command == "run":
        run(args.manifest, args.output, args.checkpoint, args.limit, args.device, args.seed)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
