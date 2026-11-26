# SENTINEL UI Animation and Workflow Fixes

## What changed
- Removed blur-based reveal transitions from the visible application surfaces and disabled backdrop blur on the top bar and UI panels.
- Increased the visibility of scroll reveals using clear vertical movement, subtle scale, and restrained orange edge-lighting.
- Added layered, independently moving orange/graphite atmosphere planes behind page content.
- Reworked Feature Intelligence into a sequential flow: all encoded input features appear first; scrolling past that stage reveals the fixed selected and excluded feature groups. The dynamic router/expert feature subsets are then explained separately.
- Changed feature chip motion to a one-shot entrance animation; it no longer strobes or loops.
- Removed the animated progress strip that travelled across the Live Simulation pipeline stage boxes. Active stages now glow/pulse in place.
- Rebuilt the Model Benchmark result area as explicit stacked rows: approach accuracy cards, then four metric tiles. Responsive breakpoints prevent overlap.
- Made Manual Prediction label the actual selected feature list by approach, explicitly calling out the predicted-family expert subset for dynamic inference.
- Emphasized the dynamic router → family-specific expert subsets in Feature Intelligence.

## Verification
- TypeScript/JSX transpilation completed with no syntax diagnostics using the globally available TypeScript compiler.
- CSS opening/closing brace counts match.
- A full Vite production build and browser interaction test were not run because dependency installation timed out in this environment.
