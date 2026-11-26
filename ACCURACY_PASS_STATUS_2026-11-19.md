# SENTINEL — accuracy pass status

Date: 2026-11-19

## Delivery status

**The requested accuracy target has not been reached.** No result in this pass supports claiming 80–90% native multiclass accuracy on both official datasets while Dynamic reliably beats both Fixed and All Features.

The updated research scripts and candidate artifacts are included in this project snapshot. The original application models and product workflows were not replaced.

## Best strictly-inductive native multiclass results from the current reproducible runs

All results below use the official test files and do not use test labels for selection or training. “Dynamic” uses only thresholds/router fits from the official-training validation partitions.

| Dataset | All Features | Fixed Features | Dynamic | Dynamic vs best baseline |
|---|---:|---:|---:|---:|
| NSL-KDD (current 100-tree base + learned correctness router) | 76.10% | 78.31% | 77.88% | -0.43 pp |
| UNSW-NB15 (confidence-margin router) | 76.33% | 76.33% | 76.34% | +0.004 pp |

The NSL-KDD Dynamic model is better than All Features but does not beat Fixed Features. The UNSW-NB15 confidence-margin Dynamic model technically edges both baselines by a negligible amount; this is not a meaningful or robust improvement and is far below 80%.

## Strongest non-adaptive research candidates

| Dataset | Candidate | Official-test accuracy | Notes |
|---|---|---:|---|
| NSL-KDD | Full-training XGBoost candidate | 76.62% | Native five-class; not Dynamic |
| UNSW-NB15 | Full-training XGBoost candidate | 76.99% | Native ten-class; not Dynamic |

## Additional experiments completed in this pass

### Filter/wrapper feature ranking

Implemented and evaluated Gain Ratio, Mutual Information, correlation ratio (eta-squared), Decision Tree impurity importance, ExtraTrees importance, and rank-fusion combinations. Ranking is performed on official-training fit data only. The compact 50% ExtraTrees-ranked subset improved some NSL-KDD model comparisons, but the Dynamic router did not reliably surpass the Fixed model.

### Unlabeled domain adaptation (transductive; separate protocol)

Unlabeled official-test feature vectors were used to train a source-versus-target domain classifier and estimate density-ratio sample weights. No test labels were used for adaptation.

| Dataset | Full model | Domain-weighted compact model | Dynamic | Finding |
|---|---:|---:|---:|---|
| NSL-KDD | 75.89% | 76.43% | 76.43% | Router selected compact model for all rows; no adaptive gain |
| UNSW-NB15 | 76.60% | 76.13% | 76.60% | Dynamic was slightly below All Features |

These transductive scores must not be presented as ordinary untouched-test inductive results.

### Self-training with unlabeled target features (transductive; separate protocol)

High-confidence pseudo-labels were generated from official-test features without reading test labels. Candidate thresholds and pseudo-label weights were chosen on training-only validation.

| Dataset | Supervised baseline | Self-trained expert | Dynamic | Finding |
|---|---:|---:|---:|---|
| NSL-KDD | 76.03% | 76.11% | 76.11% | Router selected the self-trained expert for all rows |
| UNSW-NB15 | 76.61% | 76.57% | 76.67% | Small gain over these two models; no matched Fixed baseline in this run and below the 76.99% XGBoost candidate |

### Full-training ensemble screen

A 500-estimator LightGBM, ExtraTrees and RandomForest comparison was trained on the full duplicate-grouped training fold. The selected NSL-KDD LightGBM candidate scored 75.82% on the official test. Its router selected the compact candidate for all test rows, so no Dynamic acquisition gain was demonstrated.

## Reproducibility qualification

An earlier saved report in `backend/artifacts/research/hybrid_adaptive_acquisition/nsl_kdd/learned_router_v2/results.json` records a 79.23% NSL-KDD Dynamic result. However, the exact base-model checkpoint that generated that report is no longer present: the later rerun replaced the checkpoint. Re-running the documented router using the currently saved base models produced 77.88%, not 79.23%. Therefore, 79.23% is retained as a historical experiment record only and is **not** claimed as a reproducible, shippable model result.

## Application integrity and tests

- Original active model checksums match the previously delivered project snapshot for both NSL-KDD and UNSW-NB15.
- Research candidates are stored under `backend/artifacts/research/`; active app model files were not overwritten.
- Backend regression tests: **24 passed**.
- Existing frontend, dataset management, prediction, feature intelligence, threat explorer, replay, report, history, and model-management workflows are preserved.

## Conclusion

This pass completed the requested hybrid filter/wrapper experiments and several adaptive alternatives, but it did **not** achieve the 80–90% target. Do not describe this project snapshot as having met the target. It is a reproducible research snapshot documenting both successful and failed approaches.
