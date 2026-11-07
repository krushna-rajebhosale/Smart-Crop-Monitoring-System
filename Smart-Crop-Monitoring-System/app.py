"""Smart Crop Monitoring System - Flask web application.

Workflow:  upload photo -> validate -> MobileNetV2 prediction -> SQLite advice -> result page

Run it with:   python app.py      then open  http://127.0.0.1:5000
"""

import json
import logging
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import (
    Flask, abort, flash, jsonify, redirect, render_template,
    request, send_from_directory, session, url_for,
)

import database
from image_utils import MAX_UPLOAD_BYTES, ImageValidationError, cleanup_old_uploads, validate_and_clean
from labels import condition_name, crop_name, is_healthy
from leaf_gate import LeafGate
from predictor import DiseasePredictor, PredictionError

BASE_DIR = Path(__file__).resolve().parent
SAFE_UPLOAD_NAME = re.compile(r"^[0-9a-f]{32}\.jpg$")
NOT_LEAF_MESSAGE = (
    "This photo does not look like a plant leaf, so we did not analyse it. "
    "Please upload a close-up photo of a single leaf."
)

logging.basicConfig(level=logging.INFO)


def load_secret_key(instance_path):
    """Use the SECRET_KEY environment variable, or create a random key once and reuse it."""
    from_env = os.environ.get("SECRET_KEY")
    if from_env:
        return from_env
    key_file = Path(instance_path) / "secret_key.txt"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    key_file.write_text(key, encoding="utf-8")
    return key


def load_evaluation(path):
    """Read reports/evaluation.json if you have run evaluate_model.py. Otherwise None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and "overall_accuracy" in data else None
    except (OSError, ValueError):
        return None


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    app.config.update(
        DATABASE=str(BASE_DIR / "crop_data.db"),
        UPLOAD_FOLDER=str(BASE_DIR / "uploads"),
        MODEL_PATH=str(BASE_DIR / "model" / "plant_disease_model.keras"),
        CLASS_PATH=str(BASE_DIR / "model" / "class_names.json"),
        RECOMMENDATIONS_CSV=str(BASE_DIR / "data" / "recommendations.csv"),
        EVALUATION_FILE=str(BASE_DIR / "reports" / "evaluation.json"),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        CONFIDENCE_THRESHOLD=float(os.environ.get("CONFIDENCE_THRESHOLD", "60")),
        UPLOAD_RETENTION_DAYS=int(os.environ.get("UPLOAD_RETENTION_DAYS", "7")),
        GATE_PATH=str(BASE_DIR / "model" / "leaf_gate.keras"),
        GATE_INFO_PATH=str(BASE_DIR / "model" / "leaf_gate_info.json"),
        GATE_THRESHOLD=float(os.environ.get("GATE_THRESHOLD", "0.5")),
        MIN_PLANT_COLOR=float(os.environ.get("MIN_PLANT_COLOR", "0.15")),
        RATE_LIMIT_PER_MINUTE=120,
        LOAD_MODEL=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = load_secret_key(app.instance_path)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- model + database
    predictor = DiseasePredictor(
        app.config["MODEL_PATH"], app.config["CLASS_PATH"], load=app.config["LOAD_MODEL"]
    )
    app.extensions["predictor"] = predictor
    app.extensions["leaf_gate"] = LeafGate(
        app.config["GATE_PATH"], app.config["GATE_INFO_PATH"], load=app.config["LOAD_MODEL"]
    )

    database.init_database(app.config["DATABASE"], predictor.class_names)
    csv_path = Path(app.config["RECOMMENDATIONS_CSV"])
    if csv_path.exists() and predictor.class_names:
        try:
            database.import_csv(app.config["DATABASE"], csv_path, valid_classes=set(predictor.class_names))
        except Exception:
            logging.exception("Could not import %s", csv_path.name)

    removed = cleanup_old_uploads(app.config["UPLOAD_FOLDER"], app.config["UPLOAD_RETENTION_DAYS"])
    if removed:
        logging.info("Removed %d old uploaded photo(s).", removed)

    def current_predictor():
        """Always read the predictor from the app, so tests can swap in a fake one."""
        return app.extensions["predictor"]

    def current_gate():
        return app.extensions["leaf_gate"]

    def colour_rule_value(force_leaf=False):
        """The simple colour rule is only used when the trained leaf gate is not installed."""
        if force_leaf or current_gate().ready:
            return 0
        return app.config["MIN_PLANT_COLOR"]

    def looks_like_leaf(image_path):
        """Ask the trained leaf gate (if installed). Returns True when the photo may go on."""
        gate = current_gate()
        if not gate.ready:
            return True
        probability = gate.not_leaf_probability(image_path)
        return probability < app.config["GATE_THRESHOLD"]

    # ------------------------------------------------------------------ security
    def get_csrf_token():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return session["csrf_token"]

    def csrf_is_valid():
        expected = session.get("csrf_token", "")
        sent = request.form.get("csrf_token", "")
        return bool(expected) and secrets.compare_digest(expected, sent)

    hits = defaultdict(deque)
    hits_lock = threading.Lock()

    def rate_limited():
        """Allow only a few uploads per minute from one address (protects the server)."""
        limit = app.config["RATE_LIMIT_PER_MINUTE"]
        now = time.time()
        with hits_lock:
            window = hits[request.remote_addr or "unknown"]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= limit:
                return True
            window.append(now)
        return False

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "style-src 'self' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
            "script-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        return response

    # ------------------------------------------------------------ template tools
    app.jinja_env.globals["csrf_token"] = get_csrf_token
    app.jinja_env.filters["condition"] = condition_name
    app.jinja_env.filters["crop"] = crop_name
    app.jinja_env.filters["short_time"] = lambda s: (s or "")[:16].replace("T", " ") + " UTC"

    @app.context_processor
    def inject_status():
        return {
            "model_ready": current_predictor().ready,
            "model_error": current_predictor().error,
            "class_count": len(current_predictor().class_names),
        }

    def supported_crops():
        return sorted({crop_name(n) for n in current_predictor().class_names})

    def render_index(error=None, status=200, show_override=False):
        return render_template(
            "index.html", error=error, crops=supported_crops(), show_override=show_override
        ), status

    # -------------------------------------------------------------------- routes
    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "GET":
            return render_index()

        if rate_limited():
            return render_index("Too many uploads in a short time. Please wait a minute and try again.", 429)
        if not csrf_is_valid():
            return render_index("Your session expired. Please try again.", 400)
        if not current_predictor().ready:
            return render_index(f"The analysis service is not available. {current_predictor().error}", 503)

        uploaded = request.files.get("leaf_image")
        if uploaded is None:
            return render_index("Please select a leaf image.", 400)

        # The visitor can tick "analyse anyway" if our colour check wrongly rejects a real leaf.
        force_leaf = request.form.get("force_leaf") == "1"
        try:
            saved_name, hints = validate_and_clean(
                uploaded, app.config["UPLOAD_FOLDER"], min_plant_fraction=colour_rule_value(force_leaf)
            )
        except ImageValidationError as problem:
            return render_index(str(problem), 400, show_override=(problem.code == "not_leaf"))
        except Exception:
            logging.exception("Upload handling failed")
            return render_index("An unexpected error occurred while processing the image.", 500)

        image_path = Path(app.config["UPLOAD_FOLDER"]) / saved_name
        try:
            if not force_leaf and not looks_like_leaf(image_path):
                image_path.unlink(missing_ok=True)
                return render_index(NOT_LEAF_MESSAGE, 400, show_override=True)
            result = current_predictor().predict(image_path, threshold=app.config["CONFIDENCE_THRESHOLD"])
            scan_id = database.record_scan(app.config["DATABASE"], saved_name, result)
        except PredictionError as problem:
            image_path.unlink(missing_ok=True)
            return render_index(str(problem), getattr(problem, "status", 500))
        except Exception:
            logging.exception("Analysis failed")
            image_path.unlink(missing_ok=True)
            return render_index("An unexpected error occurred while analysing the image.", 500)

        for hint in hints:
            flash(hint, "warning")
        return redirect(url_for("result", scan_id=scan_id))

    @app.route("/result/<int:scan_id>")
    def result(scan_id):
        scan = database.get_scan(app.config["DATABASE"], scan_id)
        if scan is None:
            abort(404)
        recommendation = database.get_recommendation(app.config["DATABASE"], scan["prediction"])
        image_exists = (Path(app.config["UPLOAD_FOLDER"]) / scan["image_file"]).exists()
        return render_template(
            "result.html",
            scan=scan,
            rec=recommendation,
            healthy=is_healthy(scan["prediction"]),
            image_url=url_for("uploaded_image", filename=scan["image_file"]) if image_exists else None,
            threshold=app.config["CONFIDENCE_THRESHOLD"],
        )

    @app.route("/dashboard")
    def dashboard():
        stats = database.get_stats(app.config["DATABASE"])
        max_count = max((row["n"] for row in stats["top"]), default=1)
        return render_template(
            "dashboard.html",
            stats=stats,
            max_count=max_count,
            evaluation=load_evaluation(app.config["EVALUATION_FILE"]),
            input_size=current_predictor().input_size,
            leaf_check="Trained leaf gate model" if current_gate().ready else "Simple colour rule",
            gate_info=current_gate().info if current_gate().ready else None,
        )

    @app.route("/uploads/<filename>")
    def uploaded_image(filename):
        if not SAFE_UPLOAD_NAME.match(filename):
            abort(404)
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    @app.route("/health")
    def health():
        try:
            with database.db_session(app.config["DATABASE"]) as db:
                db.execute("SELECT 1")
            db_status = "ok"
        except Exception:
            db_status = "error"
        return jsonify(
            status="ok" if current_predictor().ready and db_status == "ok" else "degraded",
            model_ready=current_predictor().ready,
            model_error=current_predictor().error,
            classes=len(current_predictor().class_names),
            leaf_check="trained_gate" if current_gate().ready else "colour_rule",
            database=db_status,
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """Simple JSON API. Nothing is saved to the history. Example (Windows PowerShell):
        curl.exe -F "leaf_image=@leaf.jpg" http://127.0.0.1:5000/api/predict
        """
        if rate_limited():
            return jsonify(error="Too many requests. Please wait a minute."), 429
        if not current_predictor().ready:
            return jsonify(error=f"The analysis service is not available. {current_predictor().error}"), 503
        uploaded = request.files.get("leaf_image")
        if uploaded is None:
            return jsonify(error="Send an image in the form field 'leaf_image'."), 400
        try:
            saved_name, hints = validate_and_clean(
                uploaded, app.config["UPLOAD_FOLDER"], min_plant_fraction=colour_rule_value()
            )
        except ImageValidationError as problem:
            return jsonify(error=str(problem), code=problem.code), 400
        image_path = Path(app.config["UPLOAD_FOLDER"]) / saved_name
        try:
            if not looks_like_leaf(image_path):
                return jsonify(error=NOT_LEAF_MESSAGE, code="not_leaf"), 400
            result = current_predictor().predict(image_path, threshold=app.config["CONFIDENCE_THRESHOLD"])
        except PredictionError as problem:
            return jsonify(error=str(problem), code=getattr(problem, "code", 500)), getattr(problem, "status", 500)
        finally:
            image_path.unlink(missing_ok=True)
        return jsonify(
            prediction=result["label"],
            crop=crop_name(result["label"]),
            condition=condition_name(result["label"]),
            confidence=result["confidence"],
            low_confidence=result["low_confidence"],
            top=result["top"],
            hints=hints,
        )

    # ------------------------------------------------------------ error handling
    @app.errorhandler(413)
    def too_large(_error):
        return render_index("That image is larger than 8 MB. Please choose a smaller one.", 413)

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", title="Page not found",
                               message="We could not find that page."), 404

    @app.errorhandler(405)
    def not_allowed(_error):
        return render_template("error.html", title="Not allowed",
                               message="That action is not allowed on this page."), 405

    @app.errorhandler(500)
    def server_error(_error):
        return render_template("error.html", title="Something went wrong",
                               message="An unexpected error occurred. Please try again."), 500

    return app


if __name__ == "__main__":
    app = create_app()
    # Debug mode is OFF by default. For learning, you may switch it on with:
    #   PowerShell:  $env:FLASK_DEBUG = "1"
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=debug)