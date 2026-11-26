# Final release verification

Date: 2026-11-19

## Completed in this repair pass

- Aligned `backend/requirements.txt` to scikit-learn **1.8.0**, matching the version used to serialize the bundled production model artifacts.
- Added Windows loky worker-count initialization before importing the ML stack to bypass the optional physical-core detection subprocess on machines without the expected Windows utility.
- Added a regression test that loads both saved production artifacts and a regression test that calls `/api/datasets`.
- Kept both original model artifacts and all existing project features/data.

## Verification results

- Backend automated suite: **31 passed**.
- Both active saved artifacts load under scikit-learn 1.8.0: NSL-KDD (5 labels) and UNSW-NB15 (10 labels).
- FastAPI smoke tests: `/api/health`, `/api/datasets`, `/api/overview`, `/api/history?limit=5`, `/api/accuracy-lab/runs` returned HTTP **200**.
- `/api/datasets` returned both datasets as trained.
- ZIP integrity check: passed.

## Not claimed

- Frontend npm installation timed out in this environment; therefore Vite production build and real-browser verification could not be completed here. Run `npm install` and `npm run build` inside `frontend` on the target machine.
- Three non-fatal test warnings remain: FastAPI `on_event` deprecation and a LightGBM feature-name warning in a unit-test fixture.
