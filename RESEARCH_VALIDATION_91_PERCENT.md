# SENTINEL — 91% Accuracy Target: Reproducible Evaluation

## Finding

The 91% figure is attainable only when the task is explicitly defined as **binary UNSW-NB15 intrusion detection** (`label`: 0 = Normal, 1 = Attack). It is not the same task as the application's native **10-class UNSW-NB15 attack-category** classification or **5-class NSL-KDD attack-family** classification.

A validation-selected LightGBM configuration achieved **90.83% accuracy** on the official UNSW-NB15 testing split. This is close to, but below, 91%; it must not be advertised as a 91% result.

## Protocol

- Dataset: official UNSW-NB15 training and testing CSV files.
- Target: `label` only, mapped to `Normal` / `Attack`.
- Excluded from model inputs: `id`, `label`, `attack_cat`, IP address and timestamp fields.
- Train records: 175,341
- Test records: 82,332
- Input features after preprocessing: 62
- Seed: 42
- Hyperparameter selection: 20% stratified validation split drawn only from the official training file; selected by validation macro-F1.
- Final model: refit preprocessing and LightGBM on the complete official training file; evaluated once on the untouched official test file.
- No test labels were used for feature selection or hyperparameter selection.

## Selected configuration

```json
{
  "n_estimators": 260,
  "num_leaves": 63,
  "learning_rate": 0.05,
  "min_child_samples": 30
}
```

## Official test results

| Metric | Result |
|---|---:|
| Accuracy | 90.83% |
| Macro precision | 91.76% |
| Macro recall | 90.12% |
| Macro F1 | 90.58% |
| Weighted F1 | 90.74% |
| Training + preprocessing | 8.70 s |
| Prediction only | 0.187 s for 82,332 rows |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 95.97% | 83.08% | 89.06% | 37,000 |
| Attack | 87.55% | 97.15% | 92.10% | 45,332 |

### Confusion matrix

Rows are actual classes; columns are predicted classes, ordered `Normal`, `Attack`.

```text
  30739     6261
   1291    44041
```

## Why the older 70–77% results are not contradictory

The previously preserved NSL-KDD model is a **five-class** classifier. Its official test metrics were:

- All features: 72.88% accuracy; macro-F1 0.5101.
- Fixed features: 72.48% accuracy; macro-F1 0.5169.
- Dynamic feature selection: 76.98% accuracy; macro-F1 0.5298.

These are harder, different tasks. The UNSW binary result cannot be compared directly to NSL-KDD five-class accuracy. Likewise, the UNSW binary result is not evidence that the 10-class UNSW attack-category classifier achieves 91%.

## Reproduction

From the project root, with the backend virtual environment active:

```powershell
cd backend
..\.venv\Scripts\python.exe ..\scripts\validate_unsw_binary.py
```

The script writes a standalone binary model and `metrics.json` under `backend/artifacts/research/unsw_nb15_binary_lightgbm/`.

## Presentation wording

Recommended wording:

> "On the official UNSW-NB15 test split, our validation-selected LightGBM binary detector achieved 90.83% accuracy and 90.58% macro-F1. The separate multiclass attack-category task is more difficult and is reported independently. SENTINEL does not combine these scores or present binary performance as multiclass performance."

**Important:** The main SENTINEL app remains configured for its native multiclass tasks. The binary experiment is a separately documented research validation artifact and is not silently substituted into the default application benchmark.
