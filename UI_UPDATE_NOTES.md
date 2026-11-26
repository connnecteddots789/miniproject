# SENTINEL UI redesign — implementation pass 1

This pass implements the black/orange visual direction and the highest-priority interaction changes from the latest UI-only master brief. Existing datasets, model artifacts, training/evaluation pipelines, simulation APIs, benchmark data, research validation, experiment persistence, and existing application routes are retained.

## Implemented in this pass

- Replaced the previous blue/cyan visual language with the restrained black, graphite, off-white, and orange SENTINEL palette.
- Updated typography, panel surfaces, buttons, borders, tables, chart containers, status indicators, and responsive spacing.
- Added original geometric SENTINEL SVG mark and wordmark assets, used the mark in the application shell and startup sequence, and added an SVG favicon.
- Refined the startup sequence with a branded geometric reveal, orange scan accent, wordmark transition, and skip control.
- Added a Command Center editorial hero section using live selected-dataset/model state.
- Refined the processing pipeline and added a persistent animation on/off preference. The pipeline remains labelled as conceptual system flow, not network topology.
- Added a training processing visual that follows the backend's actual stage field and actual reported progress. It does not invent a completion percentage or test score.
- Added a simulation processing-stage strip and a local Reset control; the existing replay events, controls, and measured event feed remain connected to the real simulation API.
- Added six separate benchmark charts using the saved benchmark results: accuracy, macro F1, macro precision/recall, training duration, inference time, and feature count.
- Rebuilt the Feature Intelligence flow as an interactive six-stage explanation using the actual saved model schema, router, fixed subset, and expert-group metadata.
- Reworked Manual Prediction to display all model input fields, separate numeric and categorical fields, populate categorical choices from the fitted encoder, require complete input, and reject invalid values before submission.
- Added backend validation for missing fields, blank values, non-finite/invalid numeric values, and unknown categorical values. Invalid requests return HTTP 422 and do not generate a prediction.
- Added regression tests for manual prediction schema and invalid-input rejection.
- Applied the SENTINEL orange/graphite identity to the PDF report header/footer, including vector brand mark, page number, and consistent accent styling.

## Verification completed

- Backend compile check: passed.
- Backend test suite: **16 passed**, 4 existing warnings.
- TypeScript/JSX transpile syntax diagnostics: **0**.
- CSS parser: **929 rules, 0 parse errors**.
- Manual prediction API smoke tests:
  - schema exposes 38 numeric and 3 categorical fields for the bundled NSL-KDD model;
  - empty input rejected with HTTP 422;
  - NaN numeric input rejected with HTTP 422;
  - invalid category rejected with HTTP 422;
  - complete valid-schema input reaches the model and returns a real prediction.
- PDF report test: passed as part of the backend suite.
- ZIP integrity: checked after packaging.

## Still pending — do not treat this pass as final completion

- Full Vite production build and browser-based visual/interaction QA were not possible because `npm install` timed out in this environment. The frontend dependencies are not installed here.
- Final visual tuning at 1920×1080, 1440×900, 1366×768, and narrow laptop widths remains pending actual browser inspection.
- Dataset Lab and other page-specific layouts still need a deeper second design pass; this iteration does not claim that every requested page has reached final presentation quality.
- The simulation inspection drawer and some of the deeper record-level interaction details should be reviewed in the next pass.

No ML methodology, training configuration, dataset split, model scores, or binary research result was changed in this UI pass.

## Implementation pass 2 — simulation investigation experience

- Reworked the simulation traffic stream into a restrained directional signal track. Each packet marker represents an actual record returned by the running dataset replay; selecting a marker opens its inspection details.
- Expanded simulation events with the JSON-safe original input record, the exact feature names used by the dynamic prediction, confidence note, and the dataset's source ground-truth label when present.
- Ground-truth labels are kept separate from predictions. Correct/incorrect is shown only where the dataset's native taxonomy can be mapped to the model's prediction taxonomy. UNSW binary `label` values are not compared to the native multi-class attack-category prediction.
- Added a simulation inspection panel showing prediction, ground truth, measured inference time, uncalibrated model score, selected features, and original record values.
- Added elapsed replay time and measured average records/second to the simulation status API and UI. The rate is calculated from the actual replay start time and processed-record count.
- Reset now stops an active/paused backend replay before clearing the local view. Stopping a replay records its finish time.
- Added a regression test for measured simulation elapsed time and throughput.

### Pass 2 verification

- Backend tests: **17 passed**, 4 warnings.
- Backend compile check: passed.
- TypeScript/JSX transpile syntax diagnostics: **0**.
- CSS brace balance: **0 difference**.
- Full Vite build and browser visual QA remain blocked until frontend dependencies can be installed. No claim of browser verification is made.

## Implementation pass 3 — research evidence presentation and compact confusion matrix

- Rebuilt the separate UNSW-NB15 binary research result as an evidence-led research panel rather than a row of generic metric cards.
- The measured **90.83% official test accuracy** is now the visual focal point, with macro-F1 (90.58%), weighted-F1 (90.74%), macro precision (91.76%) and macro recall (90.12%) shown alongside it.
- Added the experiment path: official training data → training-only stratified validation → candidate selection by validation macro-F1 → refit on the full official training split → one evaluation on the official held-out test split.
- Added the four actual validation candidate scores, selected LightGBM parameters, preprocessing summary, feature exclusions, class-weighting note, per-class metrics, measured training/inference times, and a compact binary confusion matrix. These values are loaded from `/api/research/binary/unsw`; no result values are hardcoded into the UI.
- Explicitly labels this as full 62-feature UNSW-NB15 **binary** LightGBM research. It is not presented as the native 10-class UNSW or five-class NSL-KDD result, and it is not misrepresented as the router/expert dynamic-feature result.
- Added a training-completion result summary showing accuracy, macro-F1 and weighted-F1 for All Features, Fixed Features and Dynamic Features from the selected dataset's actual benchmark response.
- Reduced the native confusion matrix cell sizing and removed the old forced 560px minimum width. It now uses a more compact square grid, with the existing raw-count/row-percentage switch and clickable cell inspection retained.
- The binary research evidence remains visible even when the selected dataset has no native trained benchmark yet.

### Pass 3 verification

- Backend tests: **17 passed**, 4 warnings.
- Python compile check: passed.
- TypeScript/JSX transpile syntax diagnostics: **0**.
- CSS parser: **0 parse errors**.
- `/api/health`: HTTP 200.
- `/api/research/binary/unsw`: HTTP 200; accuracy 0.9082738182, macro-F1 0.9058152797; confusion matrix matches saved artifact.
- Full Vite build and browser visual QA remain unverified because frontend dependencies are not installed in this environment. The npm registry install timed out in earlier passes.
- Added explicit false-negative (missed attack) and false-positive (false alarm) counts beneath the binary confusion matrix, calculated directly from its saved held-out test matrix.
