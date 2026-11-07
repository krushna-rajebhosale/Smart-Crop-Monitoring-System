"""The "Is this a leaf?" gatekeeper.

Why it exists
  Your disease model knows only 38 classes. Show it a TV or an electronic part and
  it still has to pick one of the 38, sometimes with a high score.
  This small second model looks at the photo FIRST and answers: leaf or not a leaf.
  Your disease model, its classes and class_names.json are NOT changed.

How it is created
  Train it with train_leaf_gate.py. It saves model/leaf_gate.keras.
  If that file does not exist, the website falls back to a simple colour check.
"""

import json
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image


class LeafGate:
    def __init__(self, model_path, info_path=None, load=True):
        self.model_path = Path(model_path)
        self.info_path = Path(info_path) if info_path else None
        self.model = None
        self.error = None
        self.info = None              # measured results written by train_leaf_gate.py
        self.input_size = (224, 224)  # (width, height)
        self._tf = None
        self._lock = threading.Lock()

        if self.info_path and self.info_path.exists():
            try:
                self.info = json.loads(self.info_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logging.warning("Could not read %s", self.info_path.name)

        if not load:
            self.error = "Leaf gate loading is switched off."
        elif not self.model_path.exists():
            self.error = "Leaf gate model not installed (using the simple colour check)."
        else:
            self._load()

    def _load(self):
        try:
            import tensorflow as tf
            self._tf = tf
            model = tf.keras.models.load_model(self.model_path)
            if int(model.output_shape[-1]) != 1:
                self.error = "Leaf gate model must have a single sigmoid output."
                return
            self.input_size = (int(model.input_shape[2]), int(model.input_shape[1]))
            self.model = model
            logging.info("Leaf gate model loaded.")
        except Exception:
            self.error = "Leaf gate model could not be loaded."
            logging.exception("Could not load the leaf gate model")

    @property
    def ready(self):
        return self.model is not None and self.error is None

    def not_leaf_probability(self, image_path):
        """Returns a number from 0 to 1. Close to 1 means 'not a leaf'."""
        with Image.open(image_path) as im:
            array = np.asarray(im.convert("RGB"), dtype=np.float32)
        # Same steps as in train_leaf_gate.py (Keras bilinear resize + MobileNetV2 scaling).
        width, height = self.input_size
        resized = np.asarray(self._tf.image.resize(array, (height, width)), dtype=np.float32)
        batch = self._tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        batch = np.expand_dims(batch, axis=0)
        with self._lock:
            output = self.model(batch, training=False)
        return float(np.asarray(output).reshape(-1)[0])