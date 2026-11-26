# SENTINEL — Dynamic Feature Selection Research Update

## What changed in this release

The Dynamic approach is now a two-stage, family-specialist pipeline:

1. **Shared router:** a compact multiclass model predicts the traffic family using a shared feature subset.
2. **Family specialist:** when the router predicts an attack family, that family's specialist distinguishes `Normal` from that specific family. A `Normal` router prediction exits without an expert call.
3. **Inference uses predictions only:** the router prediction determines the expert. Ground-truth test labels are never used to choose an expert.

### Training-only feature ranking

The global router/fixed ranking and each specialist's ranking use a hybrid score:

- 60% normalized ExtraTrees feature importance
- 40% normalized mutual information

The ranking is computed only from the training split (or a deterministic stratified sample of it). Preprocessing is fitted on training data only. The held-out test set is used only for final evaluation.

Each family specialist is trained on records belonging to `Normal` or that particular family. Its feature ranking is learned from that binary training subset. Specialist training uses a deterministic stratified sample capped at 12,000 records to control laptop runtime while preserving both classes.

## What to present

### Primary dataset: NSL-KDD

Use NSL-KDD for the main presentation because its standard five-family labels make the router-and-specialist idea straightforward to explain: Normal, DoS, Probe, R2L and U2R. Compare All Features, Fixed Subset and Dynamic Selection on the same official train/test split.

UNSW-NB15 is useful as a second dataset/generalization experiment. Keep its binary Normal-vs-Attack task separate from its native 10-category task and from NSL-KDD's five-family task.

### Which classifier should be selected?

There is no honest universal winner without rerunning the updated pipeline. Start with **Extra Trees** as the first candidate: it is a strong nonlinear tree ensemble and also forms the tree component of the hybrid feature-ranking method. Then compare Random Forest, XGBoost and LightGBM using the same split and seed. Select the model using validation-only tuning and a predeclared objective that considers macro-F1, accuracy, average features processed and end-to-end inference latency.

Do not claim Extra Trees is the winner until the updated run confirms it. The previously saved Decision Tree result (76.98% accuracy and 52.98% macro-F1 for Dynamic, with 54 average features versus 122 for All) belongs to the earlier pipeline and **must not be presented as the result of this updated architecture**.

## How to run a fair comparison

1. Extract this project and install the backend/frontend as described in `README.md`.
2. Open Model Training Center, choose **NSL-KDD**, choose a classifier, and run training.
3. Repeat with the same seed for each candidate classifier.
4. In Model Benchmark, compare All / Fixed / Dynamic on accuracy, macro precision, macro recall, macro-F1, weighted-F1, average features processed and end-to-end inference time.
5. Keep the same official test split and do not tune using test metrics. If you tune parameters or feature budgets, use a validation split drawn only from the training data, then run final test evaluation once.
6. Download the report after each saved run. It records the exact saved experiment; do not combine results from different classifiers or datasets into one comparison.

## How to explain the efficiency claim

Dynamic selection does not automatically mean fewer features or faster end-to-end execution. The shared router's features and the selected expert's features are both counted for each record. Report:

- average features processed per record;
- percentage reduction relative to All Features;
- end-to-end inference latency including preprocessing, router and expert;
- accuracy and macro-F1.

A defensible conclusion is that Dynamic is useful when it maintains or improves detection quality while processing fewer features. If router overhead makes it slower, report that trade-off instead of claiming a speedup.
