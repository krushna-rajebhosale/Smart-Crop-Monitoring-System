"""Tests the evaluation logic with a fake model (no TensorFlow needed)."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from evaluate_model import evaluate  # noqa: E402


class FakeModelPredictor:
    class_names = ["A___x", "B___healthy"]

    def preprocess(self, path):
        # the image colour decides the "prediction": red -> class 0, green -> class 1
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        return arr[:2, :2, :]

    def predict_probabilities(self, batch):
        out = []
        for item in batch:
            red_wins = item[..., 0].mean() > item[..., 1].mean()
            out.append([0.9, 0.1] if red_wins else [0.1, 0.9])
        return np.array(out)


class EvaluateTests(unittest.TestCase):
    def test_accuracy_and_skipped_folders(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for folder, color, n in (("A___x", (255, 0, 0), 3), ("B___healthy", (0, 255, 0), 2), ("Unknown", (0, 0, 255), 1)):
                (tmp / folder).mkdir()
                for i in range(n):
                    Image.new("RGB", (8, 8), color).save(tmp / folder / f"{i}.png")
            # one deliberately mislabelled image: green leaf stored in class A
            Image.new("RGB", (8, 8), (0, 255, 0)).save(tmp / "A___x" / "wrong.png")
            report = evaluate(FakeModelPredictor(), tmp)
            self.assertEqual(report["images_tested"], 6)
            self.assertAlmostEqual(report["overall_accuracy"], 5 / 6, places=3)
            self.assertEqual(report["folders_skipped"], ["Unknown"])
            self.assertEqual(report["top_confusions"][0]["count"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
