# Smart Crop Monitoring System

A Flask web app that detects plant leaf diseases from a photo using a MobileNetV2 model
(38 classes) and shows symptoms, prevention and cautious crop-care guidance from SQLite.

> Decision support only. Results can be wrong. Always confirm with an agricultural expert
> before applying any chemical treatment.

## Features
- Upload validation: real JPG/PNG check, size limits, corrupted-file handling, EXIF/GPS removal
- MobileNetV2 prediction with confidence, top-3 alternatives and a low-confidence warning
- Result page with symptoms, treatment, prevention, nutrients and a source badge
- Dashboard: scan history, statistics, database coverage, measured accuracy (when you run the evaluation)
- Security: CSRF token, security headers, rate limiting, random file names, no debug mode by default
- Responsive design for mobile, tablet and desktop, plus a print-friendly report
- JSON API (`POST /api/predict`) and a `/health` status page

## Requirements
- Windows 10/11, Python 3.11 (64-bit), VS Code

## Install (Windows PowerShell, inside the project folder)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Run
```powershell
python app.py
```
Open http://127.0.0.1:5000

## Test
```powershell
python -m unittest discover -s tests -v
```

## Measure real accuracy (important)
Put photos the model never saw in training into folders named exactly like the classes in
`model/class_names.json`, then run:
```powershell
python evaluate_model.py "C:\path\to\test_folder"
```
The result is saved to `reports/evaluation.json` and shown on the dashboard.

## Add checked crop-care guidance
Edit `data/recommendations.csv` (open it in VS Code). For each class fill symptoms, treatment,
prevention and nutrients ONLY from a trusted source (agricultural university, ICAR, KVK, a
government advisory), write that source in the `source` column and set `verified` to 1.
Leave a cell empty if you are not sure; the website then shows a safe general message.
Restart the app to load the changes. Edit the CSV, not the database file.

## Matching versions
The model file was saved with Keras 3.13.2. In your training notebook run
`import tensorflow as tf, keras; print(tf.__version__, keras.__version__)` and install the same
versions here if the model fails to load.

## Project structure
```
Smart-Crop-Monitoring-System/
  app.py                  routes, security, error handling
  database.py             SQLite tables, CSV import, scan history, statistics
  image_utils.py          upload validation and cleaning
  predictor.py            model loading and prediction
  labels.py               readable crop/condition names
  evaluate_model.py       measure real accuracy on a test folder
  requirements.txt
  crop_data.db            SQLite database (created/upgraded automatically)
  data/recommendations.csv
  model/plant_disease_model.keras
  model/class_names.json
  static/style.css
  static/js/app.js
  templates/ base.html _nav.html index.html result.html dashboard.html error.html
  tests/ test_app.py test_predictor.py test_evaluate.py
  reports/                evaluation.json appears here
  uploads/                temporary photos (auto-deleted after 7 days)
```

## Settings (optional environment variables)
`SECRET_KEY`, `CONFIDENCE_THRESHOLD` (default 60), `UPLOAD_RETENTION_DAYS` (default 7),
`FLASK_DEBUG=1` (learning only), `PORT`.

## Limitations
- Only the 38 trained classes are supported; other plants may be mis-identified.
- Confidence is the model's own score, not a measured accuracy.
- Models trained on clean dataset photos may perform worse on real field photos.
