"""Loads the trained MobileNetV2 model and makes predictions.

IMPORTANT: this file does NOT change your model or your class mapping.
  - same model file:      model/plant_disease_model.keras
  - same class list:      model/class_names.json  (index i = output neuron i)
  - same preprocessing:   RGB -> resize to the model input size -> mobilenet_v2.preprocess_input
It only adds safety checks and nicer error messages.
"""

import json
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image


class PredictionError(Exception):
    """The message is written for the website visitor."""


def read_class_names(class_path):
    with Path(class_path).open("r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("class_names.json must be a non-empty JSON list of text values.")
    if len(set(names)) != len(names):
        raise ValueError("class_names.json contains duplicate class names.")
    return names


class DiseasePredictor:
    def __init__(self, model_path, class_path, load=True):
        self.model_path = Path(model_path)
        self.class_path = Path(class_path)
        self.model = None
        self.class_names = []
        self.input_size = (224, 224)
        self.error = None            # short, user-friendly reason if the model is not usable
        self._tf = None
        self._lock = threading.Lock()  # Flask may serve several requests at once

        try:
            self.class_names = read_class_names(self.class_path)
        except FileNotFoundError:
            self.error = "class_names.json was not found in the model folder."
            logging.error("Missing %s", self.class_path)
        except Exception:
            self.error = "class_names.json could not be read. Check that it is a valid JSON list."
            logging.exception("Could not read class names")

        if self.error is None:
            if load:
                self._load_model()
            else:
                self.error = "Model loading is switched off."

    # ------------------------------------------------------------------ loading
    def _load_model(self):
        if not self.model_path.exists():
            self.error = f"Model file not found: model/{self.model_path.name}"
            return
        try:
            import tensorflow as tf
        except ImportError:
            self.error = "TensorFlow is not installed. Run: pip install -r requirements.txt"
            return
        try:
            self._tf = tf
            model = tf.keras.models.load_model(self.model_path)
            outputs = int(model.output_shape[-1])
            if outputs != len(self.class_names):
                self.error = (
                    f"The model has {outputs} outputs but class_names.json has "
                    f"{len(self.class_names)} classes. They must match."
                )
                logging.error(self.error)
                return
            self.input_size = (int(model.input_shape[2]), int(model.input_shape[1]))
            self.model = model
            logging.info(
                "Disease model loaded (%d classes, input %dx%d).",
                outputs, self.input_size[0], self.input_size[1],
            )
        except Exception:
            self.error = "The AI model could not be loaded. Check the TensorFlow/Keras versions."
            logging.exception("Could not load the disease model.")

    @property
    def ready(self):
        return self.model is not None and self.error is None

    # --------------------------------------------------------------- prediction
    def preprocess(self, image_path):
        """Same steps as the original project (must match the training notebook)."""
        with Image.open(image_path) as im:
            im = im.convert("RGB").resize(self.input_size)
            array = np.asarray(im, dtype=np.float32)
        return self._tf.keras.applications.mobilenet_v2.preprocess_input(array)

    def predict_probabilities(self, batch):
        """batch: array of shape (n, height, width, 3). Returns array (n, classes)."""
        if not self.ready:
            raise PredictionError(self.error or "The AI model is not available.")
        try:
            with self._lock:
                probs = self.model(batch, training=False)
            probs = np.asarray(probs)
        except Exception:
            logging.exception("Prediction failed")
            raise PredictionError("Could not analyse this image. Please try another photo.")
        if probs.shape[-1] != len(self.class_names):
            raise PredictionError("Model output count does not match class_names.json.")
        return probs

    def predict(self, image_path, top_k=3, threshold=60.0):
        """Returns a dict with the best class, confidence (%), top-k list and a low-confidence flag."""
        batch = np.expand_dims(self.preprocess(image_path), axis=0)
        probs = self.predict_probabilities(batch)[0]
        order = np.argsort(probs)[::-1][:top_k]
        top = [
            {"label": self.class_names[int(i)], "confidence": round(float(probs[int(i)]) * 100, 2)}
            for i in order
        ]
        best = top[0]
        return {
            "label": best["label"],
            "confidence": best["confidence"],
            "top": top,
            "low_confidence": best["confidence"] < threshold,
        }
