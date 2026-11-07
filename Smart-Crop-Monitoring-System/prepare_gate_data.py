"""Split your photos into train / val / test folders for train_leaf_gate.py.

Put your photos in two folders first:
    gate_raw/leaf/        leaf photos (PlantVillage photos AND real photos you took)
    gate_raw/not_leaf/    everything else (electronic parts, TVs, cars, people, rooms, soil...)

Run (from the project folder):
    python prepare_gate_data.py gate_raw gate_data

Result: gate_data/train, gate_data/val, gate_data/test (70% / 15% / 15%), each with leaf/ and not_leaf/.

WARNING about honest testing: if the same object appears many times (e.g. 20 photos of the
same circuit board, or many PlantVillage photos of one leaf), a random split puts near-copies
in both train and test and the score looks better than it really is. Best practice: keep
the photos of the SAME object together in one split, and put your own fresh phone photos
(never used for training) into gate_data/test by hand.
"""

import random
import shutil
import sys
from pathlib import Path

SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
CLASSES = ("leaf", "not_leaf")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python prepare_gate_data.py gate_raw gate_data")
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    rng = random.Random(42)
    for cls in CLASSES:
        files = sorted(p for p in (source / cls).glob("*") if p.suffix.lower() in SUFFIXES)
        if not files:
            raise SystemExit(f"No images found in {source / cls}")
        rng.shuffle(files)
        n = len(files)
        cut1, cut2 = int(n * SPLITS[0][1]), int(n * (SPLITS[0][1] + SPLITS[1][1]))
        parts = {"train": files[:cut1], "val": files[cut1:cut2], "test": files[cut2:]}
        for split, items in parts.items():
            out = target / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for path in items:
                shutil.copy2(path, out / path.name)
            print(f"{split:5s} {cls:9s} {len(items)} images")


if __name__ == "__main__":
    main()