# SENTINEL — Final UI Release Pass

## Visual direction

The application-wide UI now uses the requested original SENTINEL black/orange identity: near-black backgrounds, graphite surfaces, restrained orange actions and active states, warm neutral typography, thin borders, and compact rectangular surfaces. The reference translation follows CyberGuard's confident cybersecurity presentation and section hierarchy, while keeping the application original and research-oriented.

## Completed UI work in this release

- Original geometric SENTINEL SVG mark, wordmark and favicon.
- Branded startup sequence with skip action and reduced-motion handling.
- Responsive sidebar, collapse state, top navigation and Presentation Mode.
- Command Center hero, measured operational summaries and interactive five-stage conceptual processing pipeline.
- Pipeline animation toggle persisted locally; diagram remains explanatory, not a claimed network topology.
- Dataset Lab quality overview, class distribution, numeric feature-spread chart, missing-value inspection, source-file train/test record counts, schema inspection, upload area, and searchable paginated original-record explorer.
- Model Training workspace with actual backend stages/progress, cancellation, and measured All/Fixed/Dynamic evaluation summary after completion.
- Feature Intelligence beginner/technical explanation, interactive feature pipeline, actual importance and selected-feature views.
- Live Simulation traffic stream, controls, actual dataset events, measured throughput, and record inspection.
- Model Benchmark comparisons for accuracy, macro-F1, precision/recall, training duration, inference time and feature reduction; per-class report and compact confusion heatmap with counts/row percentages and cell inspection.
- Separate UNSW-NB15 binary research evidence panel, including validation candidate selection, configuration, official test metrics, class metrics and confusion matrix. It remains explicitly separate from the native multiclass tasks.
- Manual Prediction required-field validation in frontend and backend.
- Threat Explorer filters, sorting, record inspection and paginated results.
- Experiment history with evaluation detail, multi-run comparison, CSV export and deletion of experiment-history rows (saved model files are not deleted).
- Branded PDF report output retained.
- Responsive layouts for desktop and smaller laptop/mobile widths; reduced-motion support.

## Backend additions in this UI-focused pass

- `GET /api/inspect/{dataset}/records`: paginated original dataset rows with search, source identifier, original label and mapped ground truth.
- Dataset inspection includes per-source-file train/test record counts and numeric descriptive statistics.
- `DELETE /api/experiments/{experiment_id}` deletes only the selected experiment history row.

## Verification

- Backend pytest suite: **21 passed**.
- Python `compileall`: passed.
- TypeScript/JSX syntax transpilation: **0 syntax diagnostics**.
- PostCSS parsing: passed, zero parse errors.
- FastAPI smoke checks: health, datasets, both dataset inspections, paginated record explorer, and binary research endpoint returned HTTP 200.
- ZIP integrity: verified after packaging.

## Known verification limitation

The frontend package installation repeatedly timed out in this environment. The project has no installed `node_modules`, so a Vite production build and real browser-based interaction/visual QA could not be completed here. The source-level checks above are not a substitute for that browser test. On the user's Windows machine, run `npm install` followed by `npm run build` from `frontend/` before the presentation.

## Research integrity

The separate UNSW-NB15 binary research validation remains a full-feature binary task with its saved metrics and model artifact. It is not presented as the native 10-class UNSW-NB15 result or the five-class NSL-KDD result. Native All Features, Fixed Features and Dynamic Features results remain separate and are computed from their saved experiment artifacts.
