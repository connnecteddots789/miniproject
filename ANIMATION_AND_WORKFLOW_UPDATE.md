# SENTINEL — Depth Motion & Training Workflow Update

## UI changes
- Replaced the previous reveal-only feel with layered orange ambient fields, geometric background planes, a signal orbit on the Command Center, and a moving training rail.
- Added native scroll-linked depth transitions where supported: Command Center hero, Training Center layout, Feature Intelligence, Live Simulation traffic visualisation, and Benchmark results each have a distinct motion curve.
- Added a restrained benchmark light sweep and a separate binary-training workflow treatment.
- Reduced-motion preferences disable continuous and scroll-linked motion.
- Feature Intelligence fixed/removed feature chips now wait until the feature-list section enters the viewport, then animate in a staggered 3D sorting/reveal sequence. Re-entering the page/dataset resets the trigger.

## Workflow changes
- Moved the UNSW-NB15 binary training/evaluation action out of Model Benchmark and into Model Training Center.
- Binary task results are displayed in the Training Center only after a successful run; they remain separate from the native 10-category UNSW-NB15 task.
- Model Benchmark no longer contains the binary validation action or its historical result panel.
- The main Benchmark result area now displays held-out test accuracy for All Features, Fixed Features, and Dynamic Features together. The selected approach is outlined, and the existing per-class report, confusion matrix, and charts remain driven by that selected approach.

## Research integrity
- No test labels, metrics, or benchmark values were altered or hardcoded.
- No algorithm is declared universally superior. Classifier/feature-selection combinations must be compared on identical held-out records and reported even when dynamic selection loses.
- The 90.83% result is the separate UNSW-NB15 binary Normal-vs-Attack task, not the native 10-class task and not evidence that dynamic feature selection beats the baselines.

## Verification
- Backend regression suite: 23 passed, 4 warnings.
- TypeScript TSX transpilation syntax check: recorded in the release QA file.
- CSS parser check: recorded in the release QA file.
- Frontend dependency installation/production bundle/browser visual QA require npm dependencies and a real browser; if unavailable in the build environment, they are explicitly marked unverified.
