import importlib.util
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("summarize_results", ROOT / "scripts" / "summarize_results.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MetricSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = json.loads((ROOT / "results" / "metrics_snapshot.json").read_text(encoding="utf-8"))

    def test_snapshot_passes_validator(self):
        MODULE.validate(self.snapshot)

    def test_snapshot_matches_archived_gate_reports(self):
        pgsr = json.loads((ROOT / "results" / "raw" / "rayot" / "models" / "pgsr_control_42_7" / "results.json").read_text())["ours_30000"]
        gauge = json.loads((ROOT / "results" / "raw" / "gaugesplat" / "gaugesplat_gate.json").read_text())
        trace = json.loads((ROOT / "results" / "raw" / "tracesplat" / "tracesplat_gate.json").read_text())
        tsgs = json.loads((ROOT / "results" / "raw" / "tsgs" / "work" / "final_summary.json").read_text())

        self.assertEqual(pgsr["PSNR"], self.snapshot["published_baselines"]["PGSR"]["psnr"])
        self.assertEqual(gauge["geometry"]["candidate_chamfer"], self.snapshot["project_diagnostics"]["GaugeSplat"]["chamfer_mm"])
        self.assertEqual(trace["candidate"]["chamfer"], self.snapshot["project_diagnostics"]["TraceSplat"]["chamfer_mm"])
        self.assertEqual(tsgs["image_metrics"]["PSNR"], self.snapshot["published_baselines"]["TSGS"]["psnr"])
        self.assertEqual(tsgs["measured_chamfer"], self.snapshot["published_baselines"]["TSGS"]["chamfer_mm"])
        self.assertEqual(tsgs["decision"], "reproduction_fail")

    def test_tsgs_is_best_renderer_in_snapshot(self):
        baselines = self.snapshot["published_baselines"]
        self.assertEqual(max(baselines, key=lambda name: baselines[name]["psnr"]), "TSGS")
        self.assertEqual(max(baselines, key=lambda name: baselines[name]["ssim"]), "TSGS")
        self.assertEqual(min(baselines, key=lambda name: baselines[name]["lpips"]), "TSGS")

    def test_milo_is_best_published_geometry_baseline(self):
        baselines = self.snapshot["published_baselines"]
        self.assertEqual(min(baselines, key=lambda name: baselines[name]["chamfer_mm"]), "MILo")

    def test_first_surface_tradeoff_is_directional(self):
        baseline = self.snapshot["published_baselines"]["TSGS"]
        first = self.snapshot["project_diagnostics"]["TSGS_first_surface"]
        self.assertGreater(first["mean_d2s_mm"], baseline["mean_d2s_mm"])
        self.assertLess(first["mean_s2d_mm"], baseline["mean_s2d_mm"])
        expected = (baseline["chamfer_mm"] - first["chamfer_mm"]) / baseline["chamfer_mm"]
        self.assertTrue(math.isclose(expected, first["relative_chamfer_improvement"], abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
