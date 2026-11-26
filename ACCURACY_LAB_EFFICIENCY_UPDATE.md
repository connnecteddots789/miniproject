# Accuracy Lab — Computational Efficiency Update

Accuracy Lab now evaluates adaptive feature acquisition in addition to classification metrics.

## Adaptive execution

1. The Dynamic path first evaluates the compact, training-only selected feature subset.
2. A confidence threshold is selected on the validation partition only. The selection objective maximizes validation accuracy; ties prefer lower mean feature acquisition.
3. Records above the threshold are classified by the compact model without acquiring the remaining encoded inputs.
4. Records below the threshold acquire the remaining inputs and are sent to the full-feature fallback model.
5. The official held-out test labels are used only for final metrics.

## Efficiency metrics

Each run records:
- Encoded input feature count for All and Fixed.
- Dynamic average encoded features acquired per record.
- Dynamic fallback rate and fallback record count.
- Percentage feature reduction versus All Features.
- Wall-clock and process CPU time for model inference on the same test records.

Inference timings exclude dataset loading, preprocessing and feature ranking. They are hardware- and load-dependent. Fewer inputs do not guarantee lower latency; the report displays measured values instead of assuming a speedup. Peak memory and energy are not claimed because this pass does not instrument them.

## Reports and interface

The Accuracy Lab result panel now contains a Computational Efficiency section. PDF and CSV research reports include feature acquisition and inference timing measurements; JSON includes the complete efficiency object. The existing production model benchmark report remains separate.

## Validation

Backend test suite: 29 passed. A synthetic end-to-end Accuracy Lab worker test verifies the adaptive execution path, efficiency metrics, and PDF/CSV export generation. The frontend production build was not run because frontend dependencies are not installed in this build environment; install dependencies with `npm install` before running `npm run build` locally.
