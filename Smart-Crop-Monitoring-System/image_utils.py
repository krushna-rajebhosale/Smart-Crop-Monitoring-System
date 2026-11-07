"""Upload validation and cleaning.

Every uploaded file goes through these checks BEFORE it reaches the AI model:
  1. extension is .jpg / .jpeg / .png
  2. the file is not empty and not too big
  3. the CONTENT really is a JPEG or PNG (a renamed .txt or .gif is rejected)
  4. the picture is not corrupted, not tiny and not absurdly large
  5. the picture is re-saved as a clean JPEG with a random name
     (this removes hidden metadata such as GPS location and EXIF data)
"""

import io
import logging
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024      # keep in sync with MAX_CONTENT_LENGTH
MIN_SIDE = 64                            # pixels
MAX_PIXELS = 25_000_000                  # protects the server's memory
MIN_PLANT_FRACTION = 0.15                # at least 15% of the photo must be leaf-coloured (green/yellow)


class ImageValidationError(Exception):
    """The message is written for the website visitor.

    `code` lets the website react to special cases (for example "not_leaf").
    """

    def __init__(self, message, code=""):
        super().__init__(message)
        self.code = code


def allowed_extension(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _flatten_to_rgb(im):
    """Convert to RGB. Transparent areas become white (not black)."""
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return im.convert("RGB")


def plant_color_fraction(im):
    """Share of pixels (0 to 1) that are green or yellow-green, like leaf colours.

    This is a ROUGH colour check, not AI. It catches obvious non-leaf photos
    (cars, TVs, rooms, people) but it can be wrong, so the visitor can override it.
    """
    hsv = np.asarray(im.resize((256, 256)).convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] * 360.0 / 255.0
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0
    leafy = (hue >= 30) & (hue <= 170) & (saturation >= 0.20) & (value >= 0.15)
    return float(leafy.mean())


def quality_hints(im):
    """Soft, rough hints (NOT errors). They only help the user take a better photo."""
    hints = []
    gray = im.convert("L")
    gray.thumbnail((256, 256))
    arr = np.asarray(gray, dtype=np.float32)
    brightness = float(arr.mean())
    if brightness < 40:
        hints.append("The photo looks very dark. Try again in natural daylight.")
    elif brightness > 235:
        hints.append("The photo looks over-exposed. Try again away from direct glare.")
    if arr.shape[0] > 2 and arr.shape[1] > 2:
        centre = arr[1:-1, 1:-1]
        laplacian = (
            arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4 * centre
        )
        if float(laplacian.var()) < 25:
            hints.append("The photo may be blurry. A sharp, close-up photo gives better results.")
    return hints


def validate_and_clean(file_storage, upload_folder, min_plant_fraction=MIN_PLANT_FRACTION):
    """Validate an uploaded file. Returns (saved_filename, hints).

    Raises ImageValidationError with a friendly message if anything is wrong.
    """
    filename = file_storage.filename or ""
    if not filename:
        raise ImageValidationError("Please select a leaf image.")
    if not allowed_extension(filename):
        raise ImageValidationError("Please upload a JPG, JPEG or PNG image.")

    data = file_storage.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise ImageValidationError("That file is empty. Please choose another image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("That image is larger than 8 MB. Please choose a smaller one.")

    # Step 1: read only the header and check the real format.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            real_format = probe.format
            size = probe.size
            probe.verify()
    except Exception:
        raise ImageValidationError(
            "That file is not a readable image. Try a valid JPG or PNG photo."
        )

    if real_format not in ALLOWED_FORMATS:
        raise ImageValidationError(
            "The file content is not a real JPG or PNG image, even though the name says so."
        )
    width, height = size
    if min(width, height) < MIN_SIDE:
        raise ImageValidationError(
            f"That image is too small ({width}x{height}). Use a photo at least {MIN_SIDE}x{MIN_SIDE} pixels."
        )
    if width * height > MAX_PIXELS:
        raise ImageValidationError("That image has too many pixels. Please use a smaller photo.")

    # Step 2: decode, fix phone rotation, convert to RGB, save a clean copy.
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            clean = _flatten_to_rgb(im)
            clean.load()
    except Exception:
        logging.exception("Image decoding failed")
        raise ImageValidationError("That image could not be processed. Try another photo.")

    if min_plant_fraction > 0 and plant_color_fraction(clean) < min_plant_fraction:
        raise ImageValidationError(
            "This photo does not look like a plant leaf, so we did not analyse it. "
            "Please upload a close-up photo of a single leaf.",
            code="not_leaf",
        )

    hints = quality_hints(clean)
    saved_name = f"{uuid.uuid4().hex}.jpg"
    folder = Path(upload_folder)
    folder.mkdir(parents=True, exist_ok=True)
    clean.save(folder / saved_name, "JPEG", quality=92)
    return saved_name, hints


def cleanup_old_uploads(upload_folder, days):
    """Delete uploaded photos older than `days` days. Returns how many were removed."""
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for path in Path(upload_folder).glob("*.jpg"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            logging.warning("Could not delete old upload %s", path.name)
    return removed
