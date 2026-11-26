# SENTINEL — Research Methods

## Experimental approaches

### All-feature baseline
Uses every valid preprocessed feature supplied by the dataset pipeline.

### Fixed-feature baseline
Uses one global feature subset for all records. For the new native LightGBM research experiments, the ranking combines gain ratio, binned mutual information and LightGBM feature gain. The subset is determined from training-side data and is then fixed for every test record.

### Dynamic feature selection
Uses a compact multi-class router to propose likely labels. Only the selected top-K candidate specialists run. Each specialist is trained for its corresponding class and has its own feature subset, ranked from training-side data with class-specific mutual information. At prediction time, the system uses the router's prediction and specialist outputs; it never routes on the ground-truth test label.

The active NSL-KDD Decision Tree artifact is an earlier completed router/specialist experiment. The active native UNSW-NB15 artifact uses the newer LightGBM top-K router/specialist implementation. Their scores are comparable across approaches within each dataset, but not direct head-to-head model-family comparisons across datasets.

## Feature-ranking methods considered

- **Gain ratio:** information gain normalised by split information, which helps avoid simply preferring attributes with many distinct values.
- **Binned mutual information:** measures statistical dependence after discretising continuous values into bins; used to make the calculation tractable for larger folds and rare classes.
- **LightGBM gain:** importance based on the loss reduction attributed to feature splits in the fitted training-side booster.
- **Hybrid ranking:** combines the scaled gain-ratio, mutual-information and LightGBM-gain rankings for the global router/fixed subset. Specialist subsets use class-specific mutual information.

All ranking and preprocessing are fitted from the training split/validation data only. Test labels are reserved for final scoring.

## Validation and fitting

The newer adaptive research search reserves a validation fold from the official training data, with exact duplicate feature rows grouped so identical feature rows cannot fall on both sides of the validation boundary. Candidate configurations are selected using validation accuracy and macro-F1, then refit using the full official training split. The official test file is evaluated after model fitting.

The official training/test split is kept intact. No test records are injected into training or validation. The supplied official paired files are used; the project does not substitute a random 75/25 split when both paired files exist.

## Metrics

Accuracy, macro precision/recall/F1, weighted F1, class-level precision/recall/F1, confusion matrix, feature count and measured inference time are recorded. Macro-F1 is especially relevant because both datasets contain rare attack categories. Inference time includes the cost of a router/specialist design in the dynamic path; it is not assumed to be faster merely because fewer features are processed.

## Known research limitations

- The two active native results use different model families (Decision Tree on the saved NSL-KDD experiment, LightGBM on native UNSW-NB15). Comparisons among All, Fixed and Dynamic are within a dataset and its saved model configuration.
- The Dynamic advantage on native UNSW-NB15 is small and comes with higher inference latency.
- The separate binary experiments do not show Dynamic winning.
- A separate native NSL-KDD LightGBM trial shows All outperforming Dynamic, so Dynamic superiority is configuration-dependent.
- The project is an offline benchmark prototype; its dataset replay is not live packet capture and it does not block network traffic.
