# Release notes — Dynamic research and compact report

- Reworked Dynamic Feature Selection into a multiclass router plus family-vs-normal binary specialists.
- Replaced single Decision Tree feature ranking with training-only hybrid ranking (60% ExtraTrees importance + 40% mutual information).
- Specialist feature ranking is learned independently for each Normal-vs-family training subset.
- Router predictions, not ground-truth labels, determine specialist selection at inference.
- Preserved All Features and Fixed Subset baselines and the existing application/UI layout.
- Reduced PDF confusion-matrix heatmaps so all three matrices fit together on one page for a typical five-class report.
- Removed unnecessary forced page breaks; class-wise metrics, matrices and feature audit now flow more naturally.
- Tightened report typography and expert-feature table spacing while retaining readable warm-white pages and orange accents.
- Full backend test suite: 23 passed.
- PDF layout smoke test with a five-class sample: 4 pages (summary, class-wise results, confusion matrices, feature audit).

Important: existing saved model artifacts and old benchmark metrics were produced by the previous training architecture. Retrain after installing this release before using benchmark numbers or downloading a report. The old Decision Tree metrics are historical and are not evidence for the new architecture.
