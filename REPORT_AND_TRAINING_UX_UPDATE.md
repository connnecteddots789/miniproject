# SENTINEL report and training workflow update

- Added `SENTINEL_Project_Report.pdf`, a branded project/research report with a subtle diagonal SENTINEL watermark, project overview, dataset/task definitions, methodology, saved NSL-KDD metrics, separate UNSW-NB15 binary result, workflow and limitations.
- Updated backend PDF export styling with a branded title treatment and a subtle watermark on every page.
- Feature Intelligence now initially displays the complete encoded feature set. The user explicitly clicks **Add / sort selected features** to transition into the fixed-selected and excluded feature groups. The result can be toggled back to the complete feature space. Dynamic router/expert selection is explained as the next, separate stage.
- Moved UNSW-NB15 binary Normal-vs-Attack training into the standard Model Training Center configuration as a task choice. It uses the existing separately validated LightGBM protocol and is displayed in the same training execution panel and ordinary metric-summary style.
- The binary result remains explicitly separate from native UNSW-NB15 10-category classification and from the NSL-KDD dynamic router/expert benchmark. No binary score is substituted into the multiclass benchmark.

## Verification
- Frontend TSX transpilation: 0 diagnostics.
- Backend tests: 23 passed, 4 warnings.
- Backend PDF builder smoke test: generated a valid PDF byte stream.
- Standalone report: generated and included in the release ZIP.
- Full Vite production build and interactive browser verification were not run because frontend dependencies are not installed in this environment.
