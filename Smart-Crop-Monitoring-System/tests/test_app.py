"""Automatic tests. Run from the project folder with:

    python -m unittest discover -s tests -v

They use a FAKE predictor, so they test the website, validation and database
(not the AI model's accuracy; use evaluate_model.py for that).
"""

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import database  # noqa: E402
from app import create_app  # noqa: E402
from labels import condition_name, crop_name, is_healthy  # noqa: E402
from predictor import read_class_names  # noqa: E402

CLASSES = read_class_names(PROJECT / "model" / "class_names.json")


class FakePredictor:
    """Stands in for the real model so tests run fast and without TensorFlow."""

    def __init__(self, confidence=92.5, label="Tomato___Early_blight", ready=True):
        self.class_names = CLASSES
        self.input_size = (224, 224)
        self.ready = ready
        self.error = None if ready else "The AI model could not be loaded."
        self._confidence = confidence
        self._label = label

    def predict(self, image_path, top_k=3, threshold=60.0):
        top = [
            {"label": self._label, "confidence": self._confidence},
            {"label": "Tomato___Late_blight", "confidence": 5.0},
            {"label": "Tomato___healthy", "confidence": 1.0},
        ]
        return {
            "label": self._label,
            "confidence": self._confidence,
            "top": top,
            "low_confidence": self._confidence < threshold,
        }


def make_image(fmt="PNG", size=(120, 120), color=(40, 140, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, fmt)
    buffer.seek(0)
    return buffer


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.app = create_app({
            "TESTING": True,
            "SECRET_KEY": "test-key",
            "DATABASE": str(self.tmp / "test.db"),
            "UPLOAD_FOLDER": str(self.tmp / "uploads"),
            "EVALUATION_FILE": str(self.tmp / "none.json"),
            "LOAD_MODEL": False,
            "RATE_LIMIT_PER_MINUTE": 1000,
        })
        self.app.extensions["predictor"] = self.predictor_for_test()
        self.client = self.app.test_client()

    def predictor_for_test(self):
        return FakePredictor()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def token(self):
        self.client.get("/")
        with self.client.session_transaction() as sess:
            return sess["csrf_token"]

    def upload(self, data, filename="leaf.png", token=None, follow=False):
        return self.client.post(
            "/",
            data={"csrf_token": token or self.token(), "leaf_image": (data, filename)},
            content_type="multipart/form-data",
            follow_redirects=follow,
        )


class UploadFlowTests(AppTestCase):
    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Smart Crop Monitoring System", response.data)

    def test_valid_upload_shows_result(self):
        response = self.upload(make_image(), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Early blight", response.data)
        self.assertIn(b"92.5%", response.data)
        self.assertIn(b"Symptoms", response.data)

    def test_upload_is_saved_as_random_clean_jpeg(self):
        self.upload(make_image(), filename="../../evil name.png")
        files = list((self.tmp / "uploads").glob("*"))
        self.assertEqual(len(files), 1)
        self.assertRegex(files[0].name, r"^[0-9a-f]{32}\.jpg$")

    def test_low_confidence_shows_warning(self):
        self.app.extensions["predictor"] = FakePredictor(confidence=35.0)
        response = self.upload(make_image(), follow=True)
        self.assertIn(b"not very sure", response.data)

    def test_missing_csrf_token_is_rejected(self):
        response = self.client.post(
            "/",
            data={"leaf_image": (make_image(), "leaf.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"session expired", response.data)

    def test_no_file_selected(self):
        response = self.client.post(
            "/", data={"csrf_token": self.token()}, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"select a leaf image", response.data)

    def test_wrong_extension_rejected(self):
        response = self.upload(io.BytesIO(b"hello"), filename="notes.txt")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"JPG, JPEG or PNG", response.data)

    def test_text_file_renamed_to_jpg_rejected(self):
        response = self.upload(io.BytesIO(b"this is not an image"), filename="fake.jpg")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not a readable image", response.data)

    def test_gif_renamed_to_png_rejected(self):
        response = self.upload(make_image("GIF"), filename="sneaky.png")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not a real JPG or PNG", response.data)

    def test_tiny_image_rejected(self):
        response = self.upload(make_image(size=(20, 20)), filename="tiny.png")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"too small", response.data)

    def test_empty_file_rejected(self):
        response = self.upload(io.BytesIO(b""), filename="empty.jpg")
        self.assertEqual(response.status_code, 400)

    def test_too_large_file_gives_friendly_message(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 1000
        response = self.upload(make_image(size=(300, 300)), filename="big.png")
        self.assertEqual(response.status_code, 413)
        self.assertIn(b"larger than 8 MB", response.data)

    def test_model_not_ready_gives_friendly_message(self):
        self.app.extensions["predictor"] = FakePredictor(ready=False)
        response = self.upload(make_image())
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"not available", response.data)

    def test_non_leaf_photo_is_rejected_with_override_option(self):
        grey_room = make_image(color=(150, 140, 135))
        response = self.upload(grey_room, filename="room.png")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"does not look like a plant leaf", response.data)
        self.assertIn(b"analyse it anyway", response.data)
        self.assertEqual(list((self.tmp / "uploads").glob("*")), [])

    def test_override_allows_non_leaf_colours(self):
        response = self.client.post(
            "/",
            data={"csrf_token": self.token(), "force_leaf": "1",
                  "leaf_image": (make_image(color=(150, 140, 135)), "brown_leaf.png")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Early blight", response.data)

    def test_yellow_leaf_colours_are_accepted(self):
        response = self.upload(make_image(color=(200, 190, 60)), follow=True)
        self.assertEqual(response.status_code, 200)

    def test_api_rejects_non_leaf(self):
        response = self.client.post(
            "/api/predict",
            data={"leaf_image": (make_image(color=(150, 140, 135)), "car.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    def test_transparent_png_is_accepted(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (100, 100), (0, 128, 0, 128)).save(buffer, "PNG")
        buffer.seek(0)
        response = self.upload(buffer, filename="transparent.png", follow=True)
        self.assertEqual(response.status_code, 200)


class PagesAndSecurityTests(AppTestCase):
    def test_unknown_result_is_404(self):
        self.assertEqual(self.client.get("/result/9999").status_code, 404)

    def test_unknown_page_is_404_with_friendly_text(self):
        response = self.client.get("/nope")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"Page not found", response.data)

    def test_upload_route_blocks_bad_names(self):
        for name in ("secret.txt", "..%2Fapp.py", "abc.jpg", "x" * 32 + ".jpg"):
            self.assertEqual(self.client.get(f"/uploads/{name}").status_code, 404, name)

    def test_security_headers_present(self):
        response = self.client.get("/")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])

    def test_page_has_no_inline_scripts(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn("onchange=", html)
        self.assertNotIn("<script>", html)

    def test_dashboard_counts_scans(self):
        self.upload(make_image())
        self.upload(make_image())
        html = self.client.get("/dashboard").get_data(as_text=True)
        self.assertIn("Photos analysed", html)
        self.assertIn(">2<", html)
        self.assertIn("No accuracy is shown", html)  # nothing is invented

    def test_health_endpoint(self):
        data = self.client.get("/health").get_json()
        self.assertTrue(data["model_ready"])
        self.assertEqual(data["database"], "ok")
        self.assertEqual(data["classes"], len(CLASSES))

    def test_api_predict_returns_json_and_stores_nothing(self):
        response = self.client.post(
            "/api/predict",
            data={"leaf_image": (make_image(), "leaf.png")},
            content_type="multipart/form-data",
        )
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["prediction"], "Tomato___Early_blight")
        self.assertEqual(list((self.tmp / "uploads").glob("*")), [])

    def test_api_predict_without_file(self):
        response = self.client.post("/api/predict")
        self.assertEqual(response.status_code, 400)

    def test_rate_limit(self):
        self.app.config["RATE_LIMIT_PER_MINUTE"] = 2
        codes = [self.client.post("/api/predict").status_code for _ in range(4)]
        self.assertIn(429, codes)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_original_database_is_upgraded_without_losing_rows(self):
        db_file = self.tmp / "crop_data.db"
        shutil.copy(PROJECT / "crop_data.db", db_file)
        database.init_database(db_file, CLASSES)
        with database.db_session(db_file) as db:
            columns = {r["name"] for r in db.execute("PRAGMA table_info(recommendations)")}
            count = db.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
        self.assertTrue({"source", "verified", "updated_at"} <= columns)
        self.assertEqual(count, len(CLASSES))
        rec = database.get_recommendation(db_file, "Tomato___Early_blight")
        self.assertIn("dark leaf spots", rec["symptoms"])

    def test_every_class_has_a_safe_answer(self):
        db_file = self.tmp / "fresh.db"
        database.init_database(db_file, CLASSES)
        for name in CLASSES:
            rec = database.get_recommendation(db_file, name)
            for field in database.TEXT_FIELDS:
                self.assertTrue(rec[field].strip(), f"{name}: {field} is empty")

    def test_missing_guidance_never_invents_treatment(self):
        db_file = self.tmp / "fresh.db"
        database.init_database(db_file, CLASSES)
        rec = database.get_recommendation(db_file, "Potato___Late_blight")
        self.assertFalse(rec["has_details"])
        self.assertIn("No verified treatment", rec["treatment"])

    def test_csv_import_updates_and_keeps_existing_text(self):
        db_file = self.tmp / "fresh.db"
        database.init_database(db_file, CLASSES)
        csv_file = self.tmp / "r.csv"
        csv_file.write_text(
            "disease,crop,symptoms,treatment,prevention,nutrients,source,verified\n"
            "Potato___Late_blight,Potato,Test symptoms,,,,Example Source,1\n"
            "Not___AClass,X,a,b,c,d,,0\n",
            encoding="utf-8",
        )
        imported, skipped = database.import_csv(db_file, csv_file, set(CLASSES))
        self.assertEqual((imported, skipped), (1, ["Not___AClass"]))
        rec = database.get_recommendation(db_file, "Potato___Late_blight")
        self.assertEqual(rec["symptoms"], "Test symptoms")
        self.assertTrue(rec["verified"])
        self.assertIn("No verified treatment", rec["treatment"])  # empty cell kept the fallback

    def test_verified_requires_a_source(self):
        db_file = self.tmp / "fresh.db"
        database.init_database(db_file, CLASSES)
        csv_file = self.tmp / "r.csv"
        csv_file.write_text(
            "disease,crop,symptoms,treatment,prevention,nutrients,source,verified\n"
            "Potato___Late_blight,Potato,Some text,,,,,1\n",
            encoding="utf-8",
        )
        database.import_csv(db_file, csv_file, set(CLASSES))
        self.assertFalse(database.get_recommendation(db_file, "Potato___Late_blight")["verified"])

    def test_shipped_csv_matches_class_names(self):
        import csv
        with open(PROJECT / "data" / "recommendations.csv", encoding="utf-8-sig") as f:
            names = [row["disease"] for row in csv.DictReader(f)]
        self.assertEqual(names, CLASSES)


class LabelTests(unittest.TestCase):
    def test_label_helpers(self):
        self.assertEqual(crop_name("Pepper,_bell___Bacterial_spot"), "Pepper, bell")
        self.assertEqual(condition_name("Tomato___Early_blight"), "Early blight")
        self.assertEqual(condition_name("Tomato___healthy"), "Healthy")
        self.assertTrue(is_healthy("Apple___healthy"))
        self.assertFalse(is_healthy("Apple___Black_rot"))

    def test_class_names_are_valid(self):
        self.assertEqual(len(CLASSES), 38)
        self.assertEqual(len(set(CLASSES)), 38)


if __name__ == "__main__":
    unittest.main()
