# SENTINEL — Research-focused release notes

## What this release changes

- The proposed Dynamic Feature Selection approach is visually identified as SENTINEL's proposed method in the training result summary.
- The training summary now reports average features processed and model-only inference milliseconds per record for all three approaches, using the same held-out test set and the same inference-time definition.
- Dynamic manual prediction now applies the same model-specific feature conversion used during training/evaluation for expert inputs (important for CatBoost categorical inputs).
- The saved NSL-KDD model and its original measured metadata are preserved. A trial change to the feature-ranking estimator was rejected because its measured Dynamic results regressed; those trial artifacts are not included.
- `RESEARCH_RESULTS_AUDIT.md` records the measured result and explicitly distinguishes it from illustrative poster numbers.

## Saved NSL-KDD result

The included saved NSL-KDD five-family experiment reports Dynamic Feature Selection at 76.98% accuracy and 52.98% macro F1, above All Features (72.88%, 51.01%) and Fixed Selection (72.48%, 51.69%). Dynamic also uses 54 average features versus 122 for All Features.

This experiment does not show Dynamic beating Fixed on feature count (54 vs 42) or model-only inference time. Those measurements are displayed rather than hidden. The project therefore supports a strong claim about classification quality and reduced feature processing relative to All Features, but not a universal claim that Dynamic is fastest or uses fewer features than both baselines.

## Verification

- Backend pytest: 23 passed, 4 warnings.
- Python compileall: passed.
- TSX transpile/syntax diagnostics: 0.
- CSS/PostCSS parse: passed.
- Full Vite production build: not verified; npm dependency installation timed out in this environment.
- Real-browser UI/animation test: not run.
