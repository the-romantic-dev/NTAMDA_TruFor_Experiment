"""Run: python -m unittest discover -s tests -v (no test framework needed)."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class PipelineTests(unittest.TestCase):
    def test_pipeline_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("experiment"))

    def test_cli_has_only_baseline(self):
        import subprocess
        script = Path(__file__).resolve().parents[1] / "scripts/experiment.py"
        result = subprocess.run([sys.executable, str(script), "run", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("--conditions", result.stdout)

    def test_invalid_ground_truth_is_rejected(self):
        from experiment import measure
        with self.assertRaisesRegex(ValueError, "binary"):
            measure(np.zeros((64, 64)), np.full((64, 64), 255))

    def test_metrics_keep_official_orientation_and_separate_auc(self):
        from experiment import measure
        gt = np.zeros((64, 64), bool)
        gt[16:48, 16:48] = True
        result = measure(gt.astype(float), gt)
        inverse = measure(1.0 - gt, gt)
        self.assertEqual(result["f1_fixed"], 1)
        self.assertEqual(result["f1_best"], 1)
        self.assertEqual(inverse["f1_fixed"], 1)
        self.assertEqual(inverse["pixel_auc"], 0)
        self.assertEqual(inverse["f1_direct"], 0)
        for invalid in (np.full(gt.shape, np.nan), np.full(gt.shape, 2.0), np.zeros((3, 3))):
            with self.assertRaises(ValueError):
                measure(invalid, gt)

    def test_manifest_uses_author_table_and_rejects_missing_pairs(self):
        from experiment import prepare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fake").mkdir()
            (root / "mask").mkdir()
            gt = np.zeros((64, 64), np.uint8)
            gt[16:48, 16:48] = 255
            Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(root / "fake/x.png")
            Image.fromarray(gt).save(root / "mask/y.png")
            (root / "table.csv").write_text("fake,mask,prompt\nfake/x.png,mask/y.png,object\n")
            manifest = prepare(root, root / "manifest.json", expected=1)
            self.assertEqual(manifest["rows"][0]["id"], "x")
            self.assertEqual(manifest["rows"][0]["forged_pixels"], 1024)
            (root / "mask/y.png").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(root, root / "manifest.json", expected=1)

    def test_missing_predictions_cannot_be_reported_as_complete(self):
        import json
        from experiment import evaluate, source_hashes
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text(json.dumps({"conditions": ["clean"], "ids": ["absent"],
                                                            "dataset": "CocoGlide", "upstream_files": source_hashes()}))
            with self.assertRaises(FileNotFoundError):
                evaluate(root)

    def test_undefined_auc_does_not_remove_official_f1(self):
        from experiment import measure
        gt = np.zeros((64, 64), bool)
        gt[25:28, 25:28] = True  # erased by official 15px erosion
        result = measure(gt.astype(float), gt)
        self.assertIsNone(result["pixel_auc"])
        self.assertEqual(result["f1_best"], 0)

    def test_evaluation_counts_each_metric_and_rejects_corrupt_predictions(self):
        import json
        from experiment import evaluate, source_hashes
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw/clean").mkdir(parents=True)
            (root / "metadata.json").write_text(json.dumps({"conditions": ["clean"], "ids": ["normal", "tiny"],
                                                            "dataset": "CocoGlide", "upstream_files": source_hashes()}))
            for name, size in [("normal", 24), ("tiny", 3)]:
                gt = np.zeros((64, 64), bool)
                gt[20:20+size, 20:20+size] = True
                np.savez(root / "raw/clean" / (name + ".npz"), map=gt.astype(float), gt=gt,
                         conf=np.ones(gt.shape), seconds=1.0)
            summary = evaluate(root)[0]
            self.assertEqual(summary["n_f1_best"], 2)
            self.assertEqual(summary["n_pixel_auc"], 1)
            self.assertEqual(summary["f1_best"], .5)
            np.savez(root / "raw/clean/normal.npz", map=np.full((64, 64), np.nan), gt=gt,
                     conf=np.ones(gt.shape), seconds=1.0)
            with self.assertRaisesRegex(ValueError, "finite probabilities"):
                evaluate(root)

    def test_report_handles_undefined_diagnostics(self):
        import json
        from experiment import source_hashes
        from make_figures import make_figures
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw/clean").mkdir(parents=True)
            (root / "metadata.json").write_text(json.dumps({"conditions": ["clean"], "ids": ["large"],
                                                            "dataset": "CocoGlide", "upstream_files": source_hashes()}))
            gt = np.ones((64, 64), bool)
            gt[0, 0] = False  # pristine class disappears after dilation
            np.savez(root / "raw/clean/large.npz", map=gt.astype(float), gt=gt,
                     conf=np.zeros(gt.shape), seconds=1.0, rgb=np.zeros((64, 64, 3), np.uint8),
                     **{"np++": np.zeros(gt.shape)})
            make_figures(root, sanity_count=0)
            self.assertTrue((root / "report.md").exists())
            self.assertIn("undefined", (root / "figures/qualitative_analysis.md").read_text())


if __name__ == "__main__":
    unittest.main()
