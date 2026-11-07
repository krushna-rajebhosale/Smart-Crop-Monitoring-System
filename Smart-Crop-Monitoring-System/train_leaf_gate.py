"""Train the 'Is this a leaf?' gatekeeper model (MobileNetV2 transfer learning).

Needs folders made by prepare_gate_data.py:
    gate_data/train/{leaf,not_leaf}   gate_data/val/{leaf,not_leaf}   gate_data/test/{leaf,not_leaf}

Run (from the project folder, with the virtual environment active):
    python train_leaf_gate.py gate_data

First run downloads the small MobileNetV2 ImageNet weights (needs internet).
Output:
    model/leaf_gate.keras        the gatekeeper model
    model/leaf_gate_info.json    REAL results measured on gate_data/test (shown on the dashboard)

The model outputs ONE number = probability that the photo is NOT a leaf.
(Class order is fixed to ["leaf", "not_leaf"], so 0 = leaf and 1 = not_leaf.)
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import tensorflow as tf

keras = tf.keras
BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_ORDER = ["leaf", "not_leaf"]
THRESHOLD = 0.5


def load_split(root, name, shuffle):
    dataset = keras.utils.image_dataset_from_directory(
        Path(root) / name,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_ORDER,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )
    scale = keras.applications.mobilenet_v2.preprocess_input
    return dataset.map(lambda x, y: (scale(x), y)).prefetch(tf.data.AUTOTUNE)


def count_images(root, name):
    return {c: len([p for p in (Path(root) / name / c).glob("*") if p.is_file()]) for c in CLASS_ORDER}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_leaf_gate.py gate_data")
    root = Path(sys.argv[1])
    for split in ("train", "val", "test"):
        for cls in CLASS_ORDER:
            if not (root / split / cls).is_dir():
                raise SystemExit(f"Missing folder: {root / split / cls}")

    train = load_split(root, "train", shuffle=True)
    val = load_split(root, "val", shuffle=False)
    test = load_split(root, "test", shuffle=False)

    counts = count_images(root, "train")
    total = sum(counts.values())
    class_weight = {i: total / (2 * counts[c]) for i, c in enumerate(CLASS_ORDER)}
    print("Training images:", counts, "class weights:", class_weight)

    base = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomContrast(0.2),
    ])
    inputs = keras.Input(shape=IMAGE_SIZE + (3,))
    x = augment(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train,
        validation_data=val,
        epochs=15,
        class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # ---- honest evaluation on the TEST folder (never used for training) ----
    probabilities, labels = [], []
    for images, y in test:
        probabilities.extend(model(images, training=False).numpy().reshape(-1).tolist())
        labels.extend(y.numpy().reshape(-1).tolist())
    probabilities, labels = np.array(probabilities), np.array(labels)
    predicted_not_leaf = probabilities >= THRESHOLD
    is_not_leaf = labels == 1

    not_leaf_total = int(is_not_leaf.sum())
    leaf_total = int((~is_not_leaf).sum())
    caught = int((predicted_not_leaf & is_not_leaf).sum())
    wrongly_rejected = int((predicted_not_leaf & ~is_not_leaf).sum())
    info = {
        "date": date.today().isoformat(),
        "class_order": CLASS_ORDER,
        "output_meaning": "probability that the photo is NOT a leaf",
        "threshold": THRESHOLD,
        "train_images": counts,
        "test_images": {"leaf": leaf_total, "not_leaf": not_leaf_total},
        "test_accuracy": round(float((predicted_not_leaf == is_not_leaf).mean()), 4),
        "not_leaf_caught": round(caught / max(not_leaf_total, 1), 4),
        "leaf_wrongly_rejected": round(wrongly_rejected / max(leaf_total, 1), 4),
        "note": "Measured on gate_data/test supplied by the user. Only meaningful if those photos were not used for training.",
    }

    (BASE_DIR / "model").mkdir(exist_ok=True)
    model.save(BASE_DIR / "model" / "leaf_gate.keras")
    (BASE_DIR / "model" / "leaf_gate_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print()
    print(f"Test accuracy            : {info['test_accuracy'] * 100:.2f}%")
    print(f"Non-leaf photos caught   : {caught}/{not_leaf_total}")
    print(f"Real leaves wrongly stopped: {wrongly_rejected}/{leaf_total}")
    print("Saved model/leaf_gate.keras and model/leaf_gate_info.json")


if __name__ == "__main__":
    main()