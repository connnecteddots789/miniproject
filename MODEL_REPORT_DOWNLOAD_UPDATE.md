# SENTINEL model report download

- The existing **Model Benchmark → Download model report** action downloads a PDF generated on demand by `GET /api/report/{dataset}/pdf`.
- The PDF is built from the currently persisted multiclass model artifact for the selected dataset. It is not a generic project/research report and does not include the separate UNSW-NB15 binary validation run.
- The report contains the saved classifier, training timestamp, seed, dataset split, train/test row counts, input feature count, all/fixed/dynamic metrics, training and inference durations, class-wise precision/recall/F1, confusion matrices for all three approaches, and saved feature-selection/importance details.
- SENTINEL logo mark and subtle diagonal watermark are retained in the generated PDF.
- If metadata is not present, the report displays an unavailable value rather than inventing it.
