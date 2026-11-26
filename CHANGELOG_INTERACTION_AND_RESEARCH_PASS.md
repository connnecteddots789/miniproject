# SENTINEL — Interaction and Research Pass

## Frontend changes
- Feature Intelligence now keeps the selected/excluded feature groups visible after the reveal animation. A second click restores the full encoded feature list. The page/dataset navigation reset remains, but background feature refreshes no longer collapse the reveal.
- The selected feature group uses a slower, staggered, one-shot reveal so the user can actually see the items arrive; after the animation, the sorted groups remain visible.
- Binary training now presents four animated workflow stages and an indeterminate activity rail. It deliberately does not invent a percentage: the current binary validation worker exposes completion but not internal progress events.
- Cross-dataset research now opens a visible comparison panel, handles missing saved dataset benchmarks explicitly, and reports when there are no trained datasets rather than silently doing nothing.
- Model Benchmark now highlights dynamic features processed per record, reduction relative to the all-feature baseline, and end-to-end inference time per record using saved measured metadata.
- Header logo is centered in the main workspace, including the collapsed-sidebar state.
- Horizontal page overflow is clipped while local tables retain their own horizontal scrolling. Added a subtle orange vertical scroll-position marker.

## Research integrity
- No benchmark metrics were changed, fabricated, or selected using the held-out test labels.
- The bundled saved NSL-KDD result remains a single Decision Tree experiment: Dynamic accuracy 76.98%, Macro F1 52.98%, and 54 average features processed versus 122 for All Features. It is not proof that Dynamic is best for every classifier or resource metric.
- The binary UNSW-NB15 LightGBM experiment remains a separate Normal-vs-Attack task and is not mixed into native multiclass comparisons.
- Recommended next algorithm comparison: Decision Tree (current saved baseline), Extra Trees, Random Forest, LightGBM/XGBoost, with feature-budget and router/expert choices tuned only on a validation split from the official training file. Select by a predeclared objective combining macro-F1 and feature/inference cost, then evaluate once on the untouched official test split.

## Verification
- Backend tests: 23 passed.
- Python syntax compilation: passed.
- Frontend TypeScript parse check: no syntax diagnostics; full type/build verification unavailable because npm dependency installation timed out in this environment (React, lucide-react, and recharts are not installed here).
- Browser end-to-end verification: not performed.
