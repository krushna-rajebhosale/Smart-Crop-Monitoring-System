"""Measure the REAL accuracy of your model on a folder of test photos.

Folder layout (folder names must be exactly the class names in class_names.json):

    test_folder/
        Tomato___Early_blight/   img1.jpg  img2.jpg ...
        Tomato___healthy/        ...

Run (from the project folder):
    python evaluate_model.py "C:\\path\\to\\test_folder"

IMPORTANT: use photos the model NEVER saw during training (a held-out test set).
Testing on training photos gives a falsely high score.
The result is saved to reports/evaluation.json and shown on the dashboard.
"""

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from predictor import DiseasePredictor

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BATCH_SIZE = 32


def collect_images(folder, class_names):
    """Returns (list of (path, class_index), list of folder names that were skipped)."""
    index_of = {name: i for i, name in enumerate(class_names)}
    items, skipped = [], []
    for sub in sorted(Path(folder).iterdir()):
        if not sub.is_dir():
            continue
        if sub.name not in index_of:
            skipped.append(sub.name)
            continue
        for path in sorted(sub.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                items.append((path, index_of[sub.name]))
    return items, skipped


def evaluate(predictor, folder):
    items, skipped = collect_images(folder, predictor.class_names)
    if not items:
        raise SystemExit("No images found. Check that the sub-folder names match class_names.json.")

    predicted, actual, unreadable = [], [], 0
    for start in range(0, len(items), BATCH_SIZE):
        chunk = items[start:start + BATCH_SIZE]
        arrays, labels = [], []
        for path, label in chunk:
            try:
                arrays.append(predictor.preprocess(path))
                labels.append(label)
            except Exception:
                unreadable += 1
        if not arrays:
            continue
        probs = predictor.predict_probabilities(np.stack(arrays))
        predicted.extend(int(i) for i in probs.argmax(axis=1))
        actual.extend(labels)
        print(f"  {min(start + BATCH_SIZE, len(items))}/{len(items)} images done", end="\r")
    print()

    names = predictor.class_names
    total = len(actual)
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)

    per_class = {}
    for index, name in enumerate(names):
        class_total = sum(1 for a in actual if a == index)
        if class_total:
            class_correct = sum(1 for p, a in zip(predicted, actual) if a == index and p == index)
            per_class[name] = {"images": class_total, "accuracy": round(class_correct / class_total, 4)}

    confusions = Counter((names[a], names[p]) for p, a in zip(predicted, actual) if p != a)
    return {
        "date": date.today().isoformat(),
        "images_tested": total,
        "images_unreadable": unreadable,
        "overall_accuracy": round(correct / total, 4),
        "per_class": per_class,
        "top_confusions": [
            {"true": t, "predicted": p, "count": n} for (t, p), n in confusions.most_common(10)
        ],
        "folders_skipped": skipped,
        "note": "Measured on the folder supplied by the user. Only meaningful if those photos were not used for training.",
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python evaluate_model.py "path\\to\\test_folder"')
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    predictor = DiseasePredictor(
        BASE_DIR / "model" / "plant_disease_model.keras",
        BASE_DIR / "model" / "class_names.json",
    )
    if not predictor.ready:
        raise SystemExit(f"Model not ready: {predictor.error}")

    report = evaluate(predictor, folder)
    out_dir = BASE_DIR / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Images tested : {report['images_tested']}")
    print(f"Accuracy      : {report['overall_accuracy'] * 100:.2f}%")
    if report["folders_skipped"]:
        print("Skipped folders (name not in class_names.json):", ", ".join(report["folders_skipped"]))
    print("Saved to reports/evaluation.json")


if __name__ == "__main__":
    main()
