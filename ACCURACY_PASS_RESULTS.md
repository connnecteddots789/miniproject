# SENTINEL — Final Accuracy-Focused Pass

## What changed

This pass adds two reproducible, full-training-set XGBoost research candidates under `backend/artifacts/research/`:

- `unsw_nb15_native_xgboost_accuracy/`
- `nsl_kdd_native_xgboost_accuracy/`

The existing frontend, API routes, dataset handling, replay/simulation, threat history, reporting, original feature-intelligence views, active models, and original All/Fixed/Dynamic comparison artifacts were preserved. The new candidates are supplementary research artifacts; they do **not** silently replace the app's active models or change the task shown by the app.

## Results from this pass

Both experiments use their dataset's official train/test files, the native multiclass labels, all preprocessed input features, and an XGBoost classifier. The model is fitted on the complete official training split and evaluated on the official test split.

| Dataset / task | Approach | Accuracy | Macro-F1 | Weighted-F1 | Features |
|---|---|---:|---:|---:|---:|
| UNSW-NB15 native 10-class | XGBoost, all features | **76.99%** | 50.90% | 78.19% | 62 |
| NSL-KDD native 5-family | XGBoost, all features | 76.62% | 52.90% | 72.53% | 53 |

### UNSW-NB15 candidate selection

A grouped, training-only validation screen compared LightGBM, Extra Trees, Random Forest, XGBoost and CatBoost on the native ten-category task. XGBoost-200 produced the highest validation accuracy in that screen (80.512%). The XGBoost-200 configuration was then refit on all 175,341 official training records and evaluated on the untouched 82,332-record official test file. The held-out test accuracy was 76.99%; the validation score is not presented as test performance.

### NSL-KDD candidate

A pre-registered XGBoost configuration was fitted on all 125,973 official training records and evaluated on the 22,544-record official test file. It achieved 76.62% accuracy. This did not surpass the existing saved Dynamic Decision Tree result of 76.98%, so the existing active model was retained.

## The 91% target — task distinction matters

The project already contains a separate UNSW-NB15 binary Normal-vs-Attack experiment with 92.16% accuracy for its all-feature comparison candidate and 91.23% for its dynamic router/specialist candidate. These are **binary** results, not the native ten-category result. The independent binary LightGBM artifact reports 90.83% accuracy. None of these binary scores should be described as native multiclass accuracy.

The native multiclass test results from this pass are below 91%. This pass does not fabricate a 91% result, relabel the task, or substitute a binary score into the native multiclass benchmark.

## Important interpretation

- The new UNSW native XGBoost all-feature candidate materially improves over the previously saved native UNSW Dynamic result (76.99% vs 66.68%), but it is a different estimator/approach and does **not** prove that dynamic feature selection is superior.
- The new NSL-KDD XGBoost candidate is slightly below the existing Dynamic result (76.62% vs 76.98%).
- The original active All/Fixed/Dynamic models and all existing application functionality are retained for reproducibility and fair comparison.
- No test data was used to tune these candidate configurations. The UNSW model choice was informed by the training-only validation screen; the NSL configuration was fixed before its official-test evaluation.

## Reproduction

From the project root, with backend dependencies installed:

```bash
cd backend
python -m pytest -q
```

The supplementary fitted model bundles and their complete metrics are stored beside this document under `backend/artifacts/research/`.
