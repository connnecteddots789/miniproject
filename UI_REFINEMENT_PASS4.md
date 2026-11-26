# SENTINEL UI and workflow refinement — Pass 4

## Visual corrections
- Constrained Command Center pipeline motion to the connector track. The signal is now clipped inside a thin line between stage tiles and never overlays the stage cards.
- Removed the training progress connector pseudo-elements that crossed or sat on top of the square stage tiles.
- Rebuilt the Live Simulation packet lane as a single aligned horizontal track with contained packet markers, a thin rail, consistent packet sizing and restrained arrival motion.
- Centered the SENTINEL wordmark and mark in the application top bar on desktop.
- Stabilised Threat Explorer with a fixed-layout, horizontally scrollable table, consistent row heights, truncation and a grid-based filter toolbar.
- Reduced the confusion matrix footprint to a centered, compact SVG heatmap capped at 420px, preserving keyboard access, raw-count/row-percentage modes and cell inspection.
- Added staggered entrance motion to the actual fixed-selected and excluded feature chips.

## Manual Prediction improvements
- Added a selector for representative examples copied from the official training split for the selected dataset.
- Example rows contain only the model's real input feature schema. Target and difficulty fields are excluded.
- The datasets do not contain a gender feature; no gender field or fabricated demographic values have been added.
- Added a three-strategy what-if comparison for one validated input: All Features, Fixed Features and Dynamic Features. The comparison does not create duplicate detection-history entries.

## Benchmark and 90.83% workflow
- The main Benchmark result is now the currently saved native model's held-out test result, with the selected All/Fixed/Dynamic strategy, accuracy, Macro F1, Weighted F1, precision and recall.
- The separate UNSW-NB15 binary experiment is no longer loaded and displayed automatically.
- Added an explicit Run Binary Validation action for UNSW-NB15. It starts the reproducible training/validation/test script in a background task and only exposes the research result after the job completes successfully.
- The binary run is clearly labelled Normal-vs-Attack and remains separate from UNSW-NB15's 10-class attack-category task and NSL-KDD's 5-class task.
- Re-executed the validation script in this release workspace. It completed successfully:
  - Accuracy: 90.8274% (90.83%)
  - Macro F1: 90.5815% (90.58%)
  - Weighted F1: 90.7355% (90.74%)
  - Official held-out test records: 82,332
  - Official training records: 175,341
  - Processed feature count: 62
  - Model: LightGBM
- The 90.83% is not presented as the current multiclass model's score and is not hardcoded into the current-model summary.

## Verification
- Backend regression suite: 23 passed, 4 warnings.
- Manual example API: HTTP 200; five NSL-KDD family examples returned, each with 41 input fields and no missing values.
- Manual compare API: covered by regression test; returns all three strategy results from one validated input without writing three history events.
- Binary run API: exercised end-to-end. It returned a job ID, completed successfully, and the metrics endpoint returned 90.83% accuracy and 90.58% Macro F1.
- TypeScript/JSX transpile syntax diagnostics: 0.
- CSS/PostCSS parse: passed (1,404 top-level rules).
- ZIP integrity: checked at packaging time.
- Full Vite build and real-browser visual inspection remain unverified because frontend package installation was interrupted/timed out in this environment. Do not treat syntax checks as a substitute for browser QA.
