# SENTINEL Accuracy Lab

Accuracy Lab is a new research feature inside the existing SENTINEL portal. It is not a second website and it does not replace Model Training Center.

## Isolation guarantees

- Normal model artifacts remain under `backend/artifacts/nsl_kdd` and `backend/artifacts/unsw_nb15`.
- Accuracy Lab writes only to `backend/accuracy_lab/` and its own `accuracy_lab.db`.
- Normal training history remains in `backend/sentinel.db`; research runs are stored separately.
- Production model cache and prediction endpoints are not modified by an Accuracy Lab run.
- A shared training semaphore serializes normal model training and Accuracy Lab jobs to avoid competing large training jobs exhausting RAM/CPU.

## Using the feature

1. Open the existing SENTINEL portal and select **Accuracy Lab** in the sidebar.
2. Choose NSL-KDD or UNSW-NB15 in the Accuracy Lab's own dataset selector.
3. Choose a protocol:
   - **Random holdout:** stratified 80/20 by default, sourced from the official training file (KDDTrain+ or UNSW training-set). This is a within-training-distribution experiment, not the official test benchmark.
   - **Duplicate-aware:** exact duplicate feature rows are grouped so identical records cannot cross the train/test boundary.
   - **Official:** official train/test files. A validation partition is carved out of the official training file for threshold selection; official test labels are used only for final metrics.
4. Select Extra Trees or Random Forest, seed and fixed-feature budget, then run the comparison.
5. The engine trains All Features and Fixed Hybrid Features on the same training rows. Dynamic is a validation-tuned confidence-gated blend/fallback and may use a learned disagreement selector when validation evidence supports it.
6. Download the independent research report as PDF, CSV or JSON.

## Feature ranking

The hybrid selector combines training-only rankings from Gain Ratio (MI divided by feature entropy), Mutual Information, one-vs-rest Pearson correlation, ANOVA F-score, Chi-square, Decision Tree importance and Extra Trees importance. Rankings and preprocessing are never fitted on held-out test labels.

For predictable interactive runtime on student hardware, the model-fitting partition is capped at 30,000 rows after the holdout/validation partitions have been created. Validation and test partitions are not subsampled. The report records actual row counts and the training cap.

## Report contents

The Accuracy Lab report is separate from the existing Model Benchmark report. It includes dataset/protocol provenance, configuration, All/Fixed/Dynamic accuracy and F1 metrics, selected features, top hybrid rankings, routing policy, duplicate-overlap count, and separate confusion-matrix pages. The PDF uses landscape A4 tables with repeated headers and compact typography to avoid clipping on the 10-class UNSW-NB15 matrix.

## Interpretation and research target

The lab makes high-accuracy within-training-distribution experiments possible, but random-holdout accuracy must not be presented as official benchmark generalisation. Repeated/near-duplicate traffic and distribution shift can make the protocols behave very differently. Results are real measurements; the application does not force Dynamic to win. The research objective remains 90%+ where supportable and Dynamic outperforming both All and Fixed, but no such outcome is guaranteed.

## QA notes

- Backend regression suite: 28 tests passed.
- Accuracy Lab API exercised for run creation, polling and PDF/CSV/JSON exports.
- NSL-KDD random holdout and duplicate-aware experiments completed in QA; high-accuracy results are protocol-specific and are not official test scores.
- UNSW-NB15 random holdout completed in QA; Dynamic showed a small improvement over All and Fixed in the recorded run, while absolute accuracy remained below 80%.
- PDF outputs were parsed after generation; page dimensions, page text and multi-page confusion matrices were checked.
- Frontend TSX syntax transpilation passed. A full Vite dependency build could not be run in this environment because frontend npm dependencies were not present and package installation timed out.

### QA snapshot examples (not official benchmark results)

- NSL-KDD, random holdout, Random Forest, seed 7, 50% fixed feature budget: All 99.778%, Fixed 99.786%, Dynamic 99.786% (Dynamic tied Fixed and exceeded All by 0.008 percentage points).
- UNSW-NB15, random holdout, Extra Trees, seed 42, 50% fixed feature budget: All 77.533%, Fixed 77.550%, Dynamic 77.636% (Dynamic exceeded both by 0.086 percentage points over Fixed).

These are single-seed QA observations from the new 30,000-row training cap and are not a claim that the original official benchmark has reached 90%. Repeat seeds and protocols in Accuracy Lab and compare the complete distributions before drawing research conclusions.
