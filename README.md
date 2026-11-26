# SENTINEL — Adaptive Network Intelligence

> **Using this GitHub checkout:** The original project ZIP included benchmark datasets and saved models. This repository contains the source code. Run `python scripts/setup_datasets.py` to fetch the datasets, then use the training controls to regenerate model artifacts. The setup steps below include the dataset command.

A local academic research prototype for network intrusion detection using **all-feature**, **fixed-feature**, and **dynamic router/expert feature selection** approaches. The application has two independent environments: NSL-KDD and UNSW-NB15.

> **Bundled datasets and models:** This project ZIP includes the NSL-KDD and UNSW-NB15 training/testing pairs, the active saved model artifact for each dataset, and additional research experiment artifacts. No benchmark metric has been manually adjusted. See `FINAL_RESEARCH_RESULTS.md` for the exact test results and the limitations of each comparison.

## Startup troubleshooting

If Windows reports `No module named uvicorn`, backend dependencies were not installed successfully. From the `SENTINEL\\backend` directory run:

```powershell
..\\.venv\\Scripts\\python.exe -m pip install --upgrade --retries 5 --timeout 30 -r requirements.txt
..\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Keep that backend terminal open. In a second VS Code terminal, start the frontend from `SENTINEL\\frontend` with `npm install` (first run only), then `npm run dev`.

## What is included

- React + TypeScript + Vite interface with a spacious desktop-first cybersecurity design, navy/charcoal surfaces and cyan/teal/violet accents.
- FastAPI backend and Swagger documentation at `http://127.0.0.1:8000/docs`.
- Separate local data and model artifact directories for NSL-KDD and UNSW-NB15.
- NSL-KDD attack-family mapping: Normal, DoS, Probe, R2L, U2R.
- UNSW-NB15 native attack taxonomy: Normal, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms.
- Training-only preprocessing and feature selection; label, difficulty and known target/leakage columns are excluded from model inputs.
- All-feature, globally fixed-feature and two-stage dynamic router/expert models. Expert feature ranking uses a deterministic stratified sample (up to 2,000 rows); expert classifier fitting uses a deterministic stratified sample (up to 4,000 rows). All-feature and fixed-feature baselines use the full training split except for the explicitly capped RBF SVM (12,000 rows) and Stacking Ensemble (8,000 rows per fitted component); every approach is evaluated against the full held-out test split.
- Selectable classifiers: Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Stacking Ensemble, Linear SVM and RBF SVM.
- Saved fitted preprocessing, models, selectors, router, experts, metadata and empirical benchmark results for both active datasets; additional native and binary research variants are preserved under `backend/artifacts/research/`.
- Backend training jobs with polling, actual stage progress and timestamped logs.
- Manual and CSV batch prediction, dataset replay simulation, detection history, feature intelligence and benchmark comparison.
- SQLite detection history with stored prediction metadata and selected feature subsets, plus joblib model persistence.
- Branded SENTINEL project report PDF with a subtle watermark, plus dynamic PDF experiment reports generated from saved model results (including comparison metrics, per-class performance, confusion matrix and feature importance).
- Guided demonstration walkthrough, optional conceptual topology, interactive threat record inspection and experiment history.

## Requirements

- Python 3.12 (64-bit) recommended.
- Node.js 20+ and npm.
- Internet connection for installing Python/npm dependencies; no dataset download is required.
- Recommended: 8 GB RAM or more for full dataset training. Random Forest training time depends on CPU and dataset size.

## Quick start — Windows with VS Code

1. Extract the ZIP.
2. In VS Code, open the inner `SENTINEL` project folder (the folder containing `backend`, `frontend` and `scripts`).
3. Open the first integrated terminal at the project root and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe scripts\setup_datasets.py
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

4. Open a second VS Code terminal, from the project root, and run:

```powershell
cd frontend
npm install
npm run dev
```

5. Open **http://127.0.0.1:5173**. Keep both terminals open. The backend API documentation is at **http://127.0.0.1:8000/docs**.

Python 3.12 (64-bit) and Node.js 20+ are recommended. The datasets are already bundled and verified; no manual dataset download or upload is required.

### If you prefer separate terminals

Run the backend commands in one terminal and the frontend commands in another. Do not run `npm` commands from the `SENTINEL` root; run them from `SENTINEL\frontend`.

## Quick start — Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
python scripts/setup_datasets.py
```

Terminal 1:

```bash
cd backend
../.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173**.

## First training run

1. Start the backend and frontend.
2. Select **NSL-KDD** or **UNSW-NB15** in the left workspace selector.
3. Open **Model Training Center**.
4. Confirm the selected dataset is shown as available. If it is missing, open Dataset Lab and follow the preparation instructions.
5. Choose **Decision Tree** or **Random Forest**, set the random seed if desired, then press **Train Models**.
6. Watch backend-reported stages: loading, preprocessing, feature importance, all-feature training, fixed-feature training, router, experts, evaluation and saving.
7. On success, use Command Center, Manual Prediction, Batch Prediction, Live Simulation, Feature Intelligence, Threat Explorer and Model Benchmark.
8. Change dataset in the selector to switch to the other saved environment. No retraining occurs just because you revisit the application.

To retrain, press **Retrain Models**. To delete a dataset's model artifacts and detection history, use **Reset Experiment** and confirm.

## How to demonstrate to faculty

1. Show Dataset Lab and explain that each dataset has a separate schema and label taxonomy.
2. Train NSL-KDD with Random Forest; point out the genuine backend logs and artifact persistence.
3. Open Model Benchmark and compare all-feature, fixed-feature and dynamic-feature metrics. Explain macro-F1 versus weighted-F1 and class imbalance.
4. Open Feature Intelligence to show training-only feature importance and expert-specific subsets.
5. Run Manual Prediction and Batch Prediction; show saved detection records in Threat Explorer.
6. Run Live Simulation. It replays the local dataset test split sequentially and is **not** live packet capture.
7. Switch to UNSW-NB15, train it independently, and show its native taxonomy. If showing cross-dataset metrics, explain that the datasets have different test distributions and the raw metrics are not apples-to-apples comparisons.

## Dataset formats

### NSL-KDD

The bundled paired files are `KDDTrain+.txt` (125,973 rows) and `KDDTest+.txt` (22,544 rows), each with 43 columns: 41 network features, attack label and difficulty. The application reads these headerless files without dropping their first record, recognizes 42- and 43-column headerless files, and uses the official train/test pair when both are present. Attack labels are mapped to the five NSL-KDD families. Attack names that do not map to the supported taxonomy are excluded from this family-level experiment rather than being silently relabelled.

### UNSW-NB15

The recommended files are `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`. The `attack_cat` target is used when present, preserving UNSW's native attack categories. The binary `label` column and identifiers are excluded from the model inputs. If only a labelled CSV is available, the application uses a stratified 75/25 split. For custom files, make sure the attack-category target is named `attack_cat` or the binary target is named `label`.

## Model methodology

- **All features:** train and infer with every valid preprocessed input feature.
- **Fixed features:** rank features using a training-only tree-importance model, select a global subset and train a separate classifier.
- **Dynamic features:** a lightweight Decision Tree router predicts a family from a compact shared feature subset. The predicted family selects a specialised expert. Each expert's subset is selected using a training-only one-vs-rest family discriminator; the expert then predicts among the available family labels. To keep training practical on student hardware, each expert's feature selector uses a deterministic stratified sample of up to 2,000 training records, and each expert classifier is fitted on a deterministic stratified sample of up to 4,000 training records. The all-feature and fixed-feature baselines use the full training split except the explicitly capped RBF SVM and Stacking Ensemble configurations; every approach is evaluated on the full held-out test split. The ground-truth test label is never used at inference.
- All three approaches evaluate on the same held-out records for a given dataset. Official paired train/test files are preferred; otherwise a seeded stratified split is used.
- Metrics include accuracy, macro and weighted precision/recall/F1, per-class reports, confusion matrices, training time, inference time and feature counts.
- Dynamic feature cost counts the router subset plus the selected expert subset. Its cost and metrics are measured, not presumed to be superior.


## Research results and methodology

- `FINAL_RESEARCH_RESULTS.md` records the measured native multi-class test results, separate binary experiments, an alternative NSL-KDD LightGBM counter-result, and the claims that the evidence does and does not support.
- `RESEARCH_METHODS.md` describes the gain-ratio / binned mutual-information / LightGBM-gain ranking, the router/specialist inference path, and the grouped validation protocol.
- Do not combine the UNSW-NB15 binary Normal-vs-Attack accuracy with its native ten-category result. The binary and native tasks have different label definitions and are stored separately.
- Dynamic is ahead in the bundled native NSL-KDD Decision Tree experiment and slightly ahead in native UNSW-NB15 LightGBM, but it is not superior in every supplementary experiment and is slower on UNSW-NB15. The documented results are authoritative; do not describe an 85–90% native multi-class result because the bundled test metrics do not support that claim.

## Final release verification

- Backend automated tests: **31 passed**.
- Saved artifact compatibility: both bundled production artifacts loaded under scikit-learn 1.8.0 (NSL-KDD: 5 labels; UNSW-NB15: 10 labels).
- API smoke checks through FastAPI TestClient: `/api/health`, `/api/datasets`, `/api/overview`, `/api/history?limit=5`, and `/api/accuracy-lab/runs` all returned **200**. Both datasets report `trained: true`.
- Added regression tests for both saved-artifact loading and the `/api/datasets` endpoint.
- Added an early Windows loky worker-count default to avoid the optional physical-core probe that can fail when the Windows utility is unavailable.
- Frontend production build and browser checks were **not verified** in this environment: npm dependency installation timed out. Run `npm install` and `npm run build` in `frontend` before presentation.
- Remaining test warnings are non-fatal FastAPI lifecycle deprecation and a LightGBM feature-name warning in a unit-test fixture.
- The exact measured benchmark numbers and important counter-results are in `FINAL_RESEARCH_RESULTS.md`. That file supersedes older preliminary notes in the ZIP when values conflict.

## API endpoints

- `GET /api/health`
- `GET /api/datasets`
- `POST /api/train` and `GET /api/train/{job_id}`
- `GET /api/benchmark/{dataset}`
- `GET /api/features/{dataset}`, `GET /api/schema/{dataset}`, `GET /api/inspect/{dataset}`, `GET /api/report/{dataset}`, `GET /api/report/{dataset}/pdf`
- `POST /api/load/{dataset}`, `GET /api/experiments/{dataset}`
- `POST /api/train/{job_id}/cancel` (cooperative cancellation at stage boundaries)
- `POST /api/predict`, `POST /api/predict/batch`
- `POST /api/simulation/start`, `GET /api/simulation/{id}`, `POST /api/simulation/{id}/{pause|resume|stop}`
- `GET /api/history`, `GET /api/overview`
- `POST /api/upload/{dataset}`, `DELETE /api/reset/{dataset}`

Interactive API docs: `http://127.0.0.1:8000/docs`.

## Local persistence

- Dataset files: `backend/data/nsl_kdd/` and `backend/data/unsw_nb15/`.
- Saved models and metadata: `backend/artifacts/nsl_kdd/` and `backend/artifacts/unsw_nb15/`.
- Detection history: `backend/sentinel.db`.

These directories are intentionally not committed in the ZIP when empty; the application creates model artifacts after training. Keep them when you want to reuse your trained models.

## Tests and verification

From `backend/`:

```bash
pip install pytest
pytest -q
```

The included tests cover NSL-KDD family mapping, UNSW category preservation, target/leakage exclusion, an end-to-end synthetic fixture training run, model persistence and inference. Synthetic fixture data is used **only in automated tests**; it is not included as a dataset and must never be reported as a real NSL-KDD or UNSW benchmark.

## Known limitations

- Pre-trained active artifacts for both datasets and supplementary experiment artifacts are bundled. Use the saved-model benchmark as the current evidence; run Train Models only when you intentionally want to create a new experiment and replace the active local artifact.
- UNSW's official dataset link resolves to a SharePoint distribution that may require browser access; the setup script reports the missing files rather than claiming a successful download.
- The current app uses the official train/test pair when both conventional train/test filenames are available. Other supported CSV variants use a stratified split; verify the chosen split in the saved metadata.
- The prototype is not a production IDS, does not capture live packets, and does not block or prevent network traffic.


## Bundled NSL-KDD files

The project ZIP includes the two canonical NSL-KDD files supplied by the user:

| File | Rows | Columns | SHA-256 |
|---|---:|---:|---|
| `KDDTrain+.txt` | 125,973 | 43 | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` |
| `KDDTest+.txt` | 22,544 | 43 | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` |

The original uploaded archive also contained duplicate copies under an `nsl-kdd/` subfolder, ARFF versions, the 20-percent training subset and the KDDTest-21 subset. SENTINEL bundles only the canonical full training/test `.txt` pair to avoid duplicate ingestion and accidental mixing of alternate evaluation splits. The setup script recognises the bundled files and does not download them again.


## UI and training reliability update

- The Command Center includes an optional conceptual network topology. It is explicitly illustrative: the benchmark datasets do not contain real device topology or IP addresses. The visibility toggle is remembered for the browser session and does not affect inference.
- The Experiments page reads persisted training runs from the local SQLite database.
- Random Forest uses 100 trees, a bounded maximum depth and at most six worker processes to keep the workload practical on student laptops. Decision Trees also use bounded depth to avoid pathological tree growth on high-cardinality UNSW-NB15 data.
- Dynamic expert feature ranking uses a deterministic stratified sample capped at 2,000 training rows per family for selector fitting. Each expert classifier is fitted on a deterministic stratified sample capped at 4,000 training rows. The all-feature/fixed-feature baselines use the full training split except for the documented RBF SVM and stacking caps, and evaluation always uses the full official test split.
- Dynamic evaluation routes records using the router prediction and predicts each expert's routed records in batches. The backend reports progress while each expert batch is evaluated; it does not display fabricated progress or ETA.
- Both datasets remain isolated with separate preprocessing, schemas, artifacts, labels and benchmarks.

## Verification record

The release checks run before packaging:
- Dataset setup script: both bundled dataset pairs SHA-256 verified and reported READY.
- Backend automated tests: 4 passed.
- Python bytecode compilation: checked.
- Full Random Forest end-to-end training on both datasets and a browser-based frontend run are not claimed as completed in this packaging environment. Install dependencies and validate on the target Windows machine before the project demonstration.


## Interactive experience additions (2026-11-18)

This release preserves the existing SENTINEL training, inference, simulation, dataset, and experiment workflows and adds:

- Pause/resume for the conceptual network traffic animation; the existing show/hide preference remains session-scoped.
- A minimum model-score filter in Threat Explorer (scores are uncalibrated and records without scores remain visible).
- Click-through confusion-matrix inspection backed by real held-out test records. Each approach stores up to 25 representative test rows per actual/predicted cell in newly trained model artifacts. Existing artifacts created by older releases do not contain these samples; retrain once to enable this inspection.
- The guided demonstration now navigates through Command Center, Dataset Lab, Training, Feature Intelligence, Benchmark, Live Simulation, Threat Explorer, and Experiments.
- Page-entry motion and reduced-motion accessibility support.

The confusion-matrix record endpoint is:
`GET /api/benchmark/{dataset}/records?approach=dynamic&actual=Normal&predicted=DoS&limit=25`

The endpoint returns only samples stored from the official held-out test set. It does not use ground-truth labels to make predictions; labels are used only after inference for evaluation and inspection.


## Refinement notes (November 2026)

- Desktop UI now uses a wider workspace, larger type, expanded charts/tables, a navy/cyan/teal palette, and an optional Presentation Mode.
- Training stages reflect backend-reported stage/progress values; there is no synthetic ETA. Detailed backend logs are collapsed by default.
- Classifier choices include Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Stacking Ensemble, Linear SVM and RBF SVM. Linear SVM uses the full training split; RBF SVM uses a deterministic 12,000-row cap.
- The selected classifier is used for all/fixed models and for the dynamic router and experts. Feature ranking remains a training-only Decision Tree ranking step.
- Benchmark reports now include macro one-vs-rest false-positive rate, in addition to existing accuracy, macro/weighted precision/recall/F1, per-class reports, timing and feature count.
- XGBoost, LightGBM, CatBoost and leakage-safe stacking are implemented in the current refinement build. CatBoost uses native categorical feature indices after ordinal encoding; stacking uses stratified out-of-fold base predictions.
- Dataset replay continues to use actual local dataset rows and model predictions. It is not live packet capture.


## Extended classifier lineup (refined release)

The Model Training Center lets you select one classifier per reproducible experiment. Each run evaluates the same all-feature, fixed-feature and dynamic router/expert approaches on the same official train/test split for the selected dataset. The held-out test labels are not used for feature ranking or tuning. Runs are persisted in experiment history, so train each candidate separately to compare them.

Supported classifiers: Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Stacking Ensemble, Linear SVM and RBF SVM. XGBoost, LightGBM and CatBoost are installed from `backend/requirements.txt`. CatBoost uses a dedicated ordinal categorical preprocessing path and passes categorical feature indices to CatBoost; other classifiers use one-hot encoding for low-cardinality categories and compact ordinal encoding for high-cardinality categories. All preprocessing is fitted on the training split only.

Stacking uses Random Forest and Extra Trees, plus XGBoost when available, with a Logistic Regression meta-model. A stratified 2–3 fold `StackingClassifier` generates out-of-fold base-model predictions for meta-model fitting, rather than training the meta-model on in-sample predictions. It is computationally heavier than an individual tree model and is deterministically capped at 8,000 training rows per fitted ensemble component; the complete held-out test split is still evaluated. The report records fitted-row counts so this approximation is visible.

### SVM performance note

The default scalable SVM choice is Linear SVM (`LinearSVC`), which trains on the complete training split. RBF SVM is a separate option and uses a deterministic stratified maximum of 12,000 training records because kernel SVM training and memory costs grow rapidly with sample count. The complete held-out test split is still used for evaluation. This is an explicit training approximation for RBF SVM, not a limitation applied to every SVM. The RTX 4060 does not accelerate scikit-learn SVC automatically; the cap is due to the RBF kernel algorithm, not an assumption that the laptop is weak.

### Compute note

The implementation prioritizes stable CPU execution on Windows laptops and uses bounded CPU worker counts for tree/boosting models. An NVIDIA RTX GPU is not automatically used by these configurations; CUDA-enabled library builds and driver/toolkit compatibility would be a separate optional acceleration setup. The 13th-generation Intel Core i5 / RTX 4060 class of laptop is suitable for the supported CPU workflow, but stacking and the five-family dynamic experts can take substantially longer than a single baseline model. Keep the laptop plugged in and allow adequate cooling for longer runs.

## Dataset replay scenarios

Live Simulation is explicitly offline dataset replay. It supports sequential test-set order, deterministic randomized test-set order, and attack-focused rows selected from the dataset's own labels. Every event is still predicted by the saved model; labels select replay rows only and are stripped before prediction. This is not live packet capture.


Dynamic expert training uses a deterministic stratified sample of up to 4,000 training records per expert; expert feature ranking uses up to 2,000 records. High-cardinality categorical identifiers are ordinal-encoded to avoid dense one-hot memory blow-ups; low-cardinality categories use one-hot encoding. The router/expert models are evaluated on the complete held-out test split. The artifact stores these sample limits and per-model fitted-row counts. Reported end-to-end inference timing includes the shared preprocessing transform and the router for the dynamic path, in addition to classifier prediction time.


## Product redesign refinement (2026-11-18)

- Added a first-session SENTINEL introduction sequence with a skip action and reduced-motion support.
- Added a collapsible desktop navigation rail.
- Replaced the confusion-matrix HTML table with a scalable, keyboard-focusable SVG heatmap. Rows are actual classes; columns are predicted classes. It supports raw counts and row-normalised percentages, per-cell hover details, and click-through to stored held-out examples.
- Added Logistic Regression as an explicit selectable baseline; it uses the same training-only preprocessing and three feature strategies as the other supported classifiers.
- Preserved the existing dataset-specific pipelines, training jobs, persisted experiments, replay simulation, and report export.

### Runtime and compute notes
The backend is CPU-first. A CUDA-capable RTX 4060 is not automatically used by scikit-learn, LightGBM, XGBoost's current `hist` configuration, or the current CatBoost CPU configuration. The 13th-gen Intel i5 and 4060 laptop is suitable for this academic-scale workflow, but memory and thermal limits depend on the exact CPU, RAM, and laptop power profile. Linear SVM uses the complete training split; RBF SVM is capped to a deterministic stratified 12,000-row training sample; stacking is capped to 8,000 rows per fitted component. These limits are stored in model configuration metadata.

### Verification status
- Backend regression suite: 11 tests passed after the changes.
- Python modules compile.
- Frontend TSX transpilation/syntax check passed using the globally available TypeScript compiler.
- A full Vite production build and actual browser-based visual QA were not possible in this environment because npm dependency installation timed out. Do not treat the screenshot/structural review as proof of rendered browser correctness.


## 91% target: separate binary UNSW-NB15 research validation

The application’s default UNSW-NB15 task is **10-class attack-category classification** (`attack_cat`), while NSL-KDD is a separate **5-family classification** task. The research brief's approximately 91% target can only be compared fairly after identifying the exact task.

This release includes a separate, reproducible binary experiment using UNSW-NB15's `label` column (`0=Normal`, `1=Attack`). It selects LightGBM parameters on a validation split from the official training file, refits on the full training file, and evaluates on the untouched official test file. See `RESEARCH_VALIDATION_91_PERCENT.md` and run `python scripts/validate_unsw_binary.py` from the project root with the backend environment active.

The measured official test accuracy is **90.83%** with **90.58% macro-F1**. This is not the 10-class UNSW result and must not be presented as the NSL-KDD multiclass result. The main app's default task remains multiclass; the binary experiment is stored separately under `backend/artifacts/research/` so it does not overwrite the existing app model artifact.

## Final UI release

See [`FINAL_UI_RELEASE.md`](FINAL_UI_RELEASE.md) for the final black/orange UI pass, dataset explorer, experiment management, compact confusion matrix, evidence-led UNSW-NB15 binary research presentation, verification results and the remaining local browser-build check.

## Updated Dynamic Feature Selection research pipeline

This release uses a compact multiclass router followed by family-vs-normal binary specialists, with a training-only hybrid ExtraTrees + mutual-information feature ranking. See `DYNAMIC_FEATURE_SELECTION_RESEARCH_GUIDE.md` for the methodology, recommended presentation dataset, classifier comparison plan, and fair reporting guidance. Retrain models after updating: previously saved metrics belong to the earlier architecture.

## Accuracy Lab

The integrated **Accuracy Lab** research workspace is available in the existing portal sidebar. It has an isolated API, SQLite history, model artifacts, and PDF/CSV/JSON report exports; see [`ACCURACY_LAB_GUIDE.md`](ACCURACY_LAB_GUIDE.md) for protocols, research safeguards, measured QA examples, and report details. Accuracy Lab jobs share a training semaphore with normal Model Training to avoid concurrent heavy training runs. Research results never overwrite production models or the Model Benchmark report.
