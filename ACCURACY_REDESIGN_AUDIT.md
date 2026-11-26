# SENTINEL accuracy redesign — honest experimental audit

Date: 2026-11-19

## Requirements being pursued

- Native multiclass classification on the official NSL-KDD and UNSW-NB15 train/test files.
- Compare All Features, Fixed Feature Selection, and Dynamic Feature Selection on identical outer test rows.
- Select feature rankings, model families, thresholds, and hyperparameters using training-only validation data.
- Preserve the existing application and its features; experimental candidates are kept separate unless they pass verification.
- Do not relabel binary results as native multiclass results and do not claim a target that was not measured.

## Current official-test evidence

| Dataset / task | All Features | Fixed | Dynamic | Strongest additional candidate |
|---|---:|---:|---:|---:|
| NSL-KDD native 5-family | 78.535% accuracy | 76.477% | 78.535% | XGBoost all-feature: 76.619% |
| UNSW-NB15 native 10-category | 66.282% accuracy | 66.073% | 66.682% | XGBoost all-feature: 76.991% |

The NSL-KDD adaptive run used a confidence-gated compact-to-full cascade. It tied All Features on the official test, rather than beating it. The existing UNSW adaptive specialist artifact has a small measured lead over All Features (+0.400 percentage points), but it is nowhere near the requested 85–90% range. The all-feature XGBoost candidate is stronger than the existing UNSW All/Fixed/Dynamic baselines, but it is not an adaptive result and must not be presented as one.

## Model-family screen (training-only validation, not test)

On NSL-KDD native five-family, a group-separated validation screen found:

- LightGBM-300: 99.688% accuracy, 89.335% macro-F1
- ExtraTrees-180: 99.492% accuracy, 86.626% macro-F1
- RandomForest-180: 99.508% accuracy, 91.116% macro-F1
- XGBoost-200: 99.645% accuracy, 88.583% macro-F1
- CatBoost-200: 98.979% accuracy, 80.213% macro-F1

These validation numbers do not transfer to the official test distribution and must not be advertised as test performance.

## Dataset-shift diagnostic

The official NSL-KDD training and test splits have materially different class proportions, particularly R2L:

| Class | Train | Test |
|---|---:|---:|
| Normal | 53.46% | 43.08% |
| DoS | 36.46% | 33.09% |
| Probe | 9.25% | 10.74% |
| R2L | 0.79% | 12.80% |
| U2R | 0.04% | 0.30% |

This is a severe train/test prior and attack-family shift. The training split contains only 995 R2L and 52 U2R records, while the official test has 2,885 R2L and 67 U2R. A random validation split from training therefore gives an overly optimistic estimate for those rare families. A model reaching 99% on this validation split does not imply 99% on the official test.

UNSW-NB15 also has class-prior differences, though less extreme for the largest classes. Rare classes remain very small (130 Worms train records and 44 test records).

## Execution status and safeguards

- The adaptive feature-selection script performs training-only feature ranking using gain ratio, binned mutual information, and LightGBM gain; it also tests router/specialist and confidence-gated cascades.
- A full native NSL-KDD candidate was refit and evaluated on the official test. It did not beat All Features.
- The UNSW native selection stage completed, but its LightGBM multi-class refit worker did not complete within the execution window. No final dynamic result is claimed for that attempted run.
- The active application model files were restored from the previously packaged project after the NSL-KDD research-only test, so an unverified candidate was not left installed in the app.
- Backend regression suite: 24 passed.

## Decision

The requested 85–90% native-multiclass result with Dynamic clearly ahead of both baselines has **not** been achieved. The evidence points to a genuine generalisation problem, especially NSL-KDD's R2L distribution shift and tiny U2R training support. Do not ship a claim that the target was met. Further work should prioritize rare-family training coverage and robust distribution-shift evaluation, then compare dynamic acquisition against the strongest non-adaptive model using the same official test rows.

## Continuation update — 19 November 2026

A further pass evaluated training-only Gain Ratio / MI / eta-squared / Decision Tree / ExtraTrees rank fusion; learned correctness routing; full-training forest/boosting; unlabeled covariate-shift weighting; and self-training on unlabeled target features. Full numerical results and protocol qualifications are recorded in `ACCURACY_PASS_STATUS_2026-11-19.md`.

The requested 80–90% native multiclass target remains unmet. The current saved strict-inductive Dynamic result for NSL-KDD (77.88%) does not beat the 78.31% Fixed baseline. UNSW-NB15 Dynamic reaches 76.34% versus 76.33% for both baselines, a negligible difference and far below target. A historical 79.23% NSL-KDD result could not be reproduced with the currently saved base-model checkpoint and is not claimed as a verified deliverable.

The app's active model files were checksum-verified against the previously delivered project snapshot and are unchanged. Backend regression tests pass: 24/24.
