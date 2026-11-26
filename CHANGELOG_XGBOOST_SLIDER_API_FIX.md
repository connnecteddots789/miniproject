# SENTINEL XGBoost, slider and API reliability fix

## XGBoost multiclass / binary class-count fix
- `LabelEncodedClassifier.fit` now determines the number of classes from the actual encoded training target on every fit.
- For two-class specialist fits it sets `objective=binary:logistic` and `eval_metric=logloss`.
- For multiclass router and approach fits it sets `objective=multi:softprob`, `num_class=len(classes)`, and `eval_metric=mlogloss`.
- The adapter rejects one-class training data with a clear message rather than allowing an invalid XGBoost configuration.
- This covers the shared multiclass router and the family-vs-Normal binary specialists.

## Benchmark trade-off transparency
- The training result summary now explicitly compares Dynamic's average feature count with Fixed's count and shows the percentage reduction.
- It also reports Dynamic's accuracy gap against the best of All and Fixed, in percentage points.
- A note clarifies that feature count is only a proxy for input-processing cost; it is not a direct CPU-energy measurement, and router/specialist overhead must be considered through end-to-end timing.
- No accuracy, feature count, or timing values are fabricated or adjusted.

## Orange slider
- Added explicit WebKit and Firefox range-track and thumb styling, including focus-visible styling.

## Local API / failed fetch diagnostics
- Frontend API base remains `http://127.0.0.1:8000/api` by default and can be overridden with `VITE_API_BASE_URL`.
- Backend CORS now accepts localhost and 127.0.0.1 development origins on any Vite port, including when Vite moves from 5173 to 5174.
- Frontend API errors now distinguish a network/backend connection failure from an HTTP/API error and provide the correct local backend troubleshooting hint.

## Tests
- Backend test suite: 24 passed.
- Added an XGBoost adapter test that fits both four-class and two-class targets and checks objective, class count and probability dimensions.
- The XGBoost smoke test was executed with the installed XGBoost package.
