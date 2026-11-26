# Release Verification

This release preserves the SENTINEL datasets, saved model artifacts, research validation, prediction history and experiment database from the previous checkpoint.

Verified in this environment:
- 21 backend tests passed.
- Python compilation passed.
- TypeScript/JSX syntax transpilation produced zero diagnostics.
- PostCSS parsed the complete stylesheet without errors.
- FastAPI smoke checks passed for dataset health, inspection, paginated original-record browsing and the separate UNSW binary research metrics endpoint.
- Final ZIP passed CRC integrity verification.

Not verified in this environment:
- Vite production build.
- Actual Chromium/Playwright navigation and visual inspection at 1920×1080, 1440×900, 1366×768 and narrow laptop widths.

Reason: `npm install --no-audit --no-fund` timed out. Install frontend dependencies locally, then run `npm run build` and launch the app for final browser inspection.
