# SENTINEL delivery status — 19 November 2026

This package preserves the existing SENTINEL application and includes the accuracy research artifacts, candidate models, scripts, and experiment audit.

## Verification

- Backend regression suite: **24 passed** (`cd backend && python -m pytest -q`).
- Existing application model artifacts were preserved; experimental candidates remain in `backend/artifacts/research/` and do not silently replace production models.
- Official train/test splits were used for the reported test metrics. Model selection was confined to training-side validation.

## Native multiclass results (official test)

| Dataset | All Features | Fixed Features | Dynamic Features | Additional candidate |
|---|---:|---:|---:|---:|
| NSL-KDD (5 classes) | 78.535% | 76.477% | 78.535% | XGBoost all-feature: 76.619% |
| UNSW-NB15 (10 categories) | 66.282% | 66.073% | 66.682% | XGBoost all-feature: 76.991% |

The requested 85–90% native-multiclass target with Dynamic clearly beating both baselines has **not** been achieved. The results are not represented as meeting that target. The 92.163% UNSW binary result is a separate Normal-vs-Attack task and is not a native 10-category result.

## Research limitation

NSL-KDD has a substantial official train/test class-prior shift, especially R2L (0.79% of training rows versus 12.80% of test rows). The audit and detailed per-class metrics are included so further work can focus on this generalisation issue without contaminating the test set.
