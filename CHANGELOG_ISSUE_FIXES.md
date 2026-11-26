# SENTINEL issue-fix release

This update addresses the requested interaction and reporting changes:

- Feature Intelligence now remounts the selected/excluded feature lists for each reveal and uses a single deterministic arrival animation so the button does not appear to start and then reset.
- The UNSW-NB15 binary LightGBM evaluation is surfaced in Model Benchmark as its own clearly labelled binary task, separately from the native 10-category benchmark. Saved binary metrics are reloaded when UNSW-NB15 is selected.
- Binary report download is generated on demand from the saved binary metrics and includes summary metrics, confusion matrix, and per-class results. It is not mixed into the native multiclass report.
- Binary training now uses an indeterminate activity rail rather than displaying a misleading 0% progress figure while the current script runs without granular progress callbacks.
- Binary training cancellation now terminates the child process when cancellation is requested. The standard training cancellation endpoint remains available.
- Command Center adds bars for the attack categories present in recorded malicious predictions. Counts are explicitly described as predictions from the local event log, not ground truth.
- The centered topbar SENTINEL brand is offset to the center of the main workspace, accounting for the sidebar.
- Removed residual blur from the scroll-reveal and feature-chip elements.

## Verification

- Python syntax compilation: passed.
- Backend focused tests: 6 passed (`test_reporting`, `test_experiment_delete`, `test_manual_prediction_validation`).
- Binary PDF generation smoke test from the bundled saved metrics: passed.
- Frontend TypeScript parse check: no JSX/syntax diagnostics; dependency-resolution diagnostics remain because frontend packages were not installed in this environment.
- Full frontend production build and browser end-to-end test were not run.


## Follow-up polish
- Feature Intelligence now presents fixed selected features in descending saved training-only importance order and excluded features alphabetically; each chip receives a staggered one-time reveal. The reveal is positional/opacity-only and does not use blur.
- PDF reports use a subtle warm-white page with restrained orange header tint and edge accent, more efficient pagination, wrapped feature-subset content, and vector heatmap confusion matrices instead of table-shaped matrices.
- Binary PDF confusion matrix uses the same heatmap visual treatment.
