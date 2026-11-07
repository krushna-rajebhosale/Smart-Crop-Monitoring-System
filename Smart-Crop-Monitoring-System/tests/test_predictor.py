"""Tests the predictor's own logic using a FAKE 'tensorflow' module.

This checks our code (class-count check, top-3, low-confidence flag, friendly errors).
It does NOT prove the real model is accurate: use evaluate_model.py for that.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from predictor import DiseasePredictor, PredictionError  # noqa: E402

CLASSES = ["A___x", "A___healthy", "B___y"]


class FakeModel:
    def __init__(self, outputs, probs):
        self.output_shape = (None, outputs)
        self.input_shape = (None, 224, 224, 3)
        self._probs = np.array([probs], dtype=np.float32)

    def __call__(self, batch, training=False):
        return np.repeat(self._probs, len(batch), axis=0)


def install_fake_tensorflow(model):
    tf = types.ModuleType("tensorflow")
    tf.keras = types.SimpleNamespace(
        models=types.SimpleNamespace(load_model=lambda path: model),
        applications=types.SimpleNamespace(
            mobilenet_v2=types.SimpleNamespace(preprocess_input=lambda x: x / 127.5 - 1.0)
        ),
    )
    sys.modules["tensorflow"] = tf


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "m.keras").write_bytes(b"x")
        import json
        (self.tmp / "c.json").write_text(json.dumps(CLASSES), encoding="utf-8")
        self.image = self.tmp / "leaf.jpg"
        Image.new("RGB", (300, 200), (30, 120, 40)).save(self.image)

    def tearDown(self):
        sys.modules.pop("tensorflow", None)

    def test_predict_top3_and_confidence(self):
        install_fake_tensorflow(FakeModel(3, [0.1, 0.7, 0.2]))
        p = DiseasePredictor(self.tmp / "m.keras", self.tmp / "c.json")
        self.assertTrue(p.ready)
        result = p.predict(self.image, threshold=60)
        self.assertEqual(result["label"], "A___healthy")
        self.assertEqual(result["confidence"], 70.0)
        self.assertEqual([t["label"] for t in result["top"]], ["A___healthy", "B___y", "A___x"])
        self.assertFalse(result["low_confidence"])

    def test_low_confidence_flag(self):
        install_fake_tensorflow(FakeModel(3, [0.4, 0.35, 0.25]))
        p = DiseasePredictor(self.tmp / "m.keras", self.tmp / "c.json")
        self.assertTrue(p.predict(self.image, threshold=60)["low_confidence"])

    def test_class_count_mismatch_is_reported(self):
        install_fake_tensorflow(FakeModel(5, [0.2] * 5))
        p = DiseasePredictor(self.tmp / "m.keras", self.tmp / "c.json")
        self.assertFalse(p.ready)
        self.assertIn("must match", p.error)
        with self.assertRaises(PredictionError):
            p.predict_probabilities(np.zeros((1, 224, 224, 3), dtype=np.float32))

    def test_missing_model_file(self):
        p = DiseasePredictor(self.tmp / "nope.keras", self.tmp / "c.json")
        self.assertFalse(p.ready)
        self.assertIn("Model file not found", p.error)
        self.assertEqual(p.class_names, CLASSES)  # class names still readable

    def test_bad_class_file(self):
        (self.tmp / "bad.json").write_text('{"a": 1}', encoding="utf-8")
        p = DiseasePredictor(self.tmp / "m.keras", self.tmp / "bad.json")
        self.assertFalse(p.ready)
        self.assertIn("class_names.json", p.error)

    def test_duplicate_class_names_rejected(self):
        (self.tmp / "dup.json").write_text('["a", "a"]', encoding="utf-8")
        p = DiseasePredictor(self.tmp / "m.keras", self.tmp / "dup.json")
        self.assertFalse(p.ready)


if __name__ == "__main__":
    unittest.main()
