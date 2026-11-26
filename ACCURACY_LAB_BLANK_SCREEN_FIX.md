# Accuracy Lab blank-screen fix

## Root cause

The Accuracy Lab backend creates each new run with `status='queued'` and `summary={}`. The frontend previously treated any truthy `run.summary` as completed results and immediately dereferenced `summary.results.all_features`. During the first status poll, this caused a React render exception and blanked the entire application before the experiment had finished.

## Changes

- Render result cards, metrics, confusion matrices and reports only after the run status is `completed` and all three result payloads plus efficiency metrics are present.
- Harden status polling against non-JSON and non-200 responses; stop the busy state for completed, failed or cancelled terminal states.
- Reset the prior run ID when starting a new experiment.
- Add an application-level React error boundary with a recovery screen so an unexpected UI exception no longer leaves the user with an unrecoverable blank page.
- Bound Accuracy Lab ensemble parallelism to two workers by default (configurable with `SENTINEL_ML_WORKERS`, capped at four) rather than using all available logical cores. Set `LOKY_MAX_CPU_COUNT` before sklearn/joblib initialization to avoid Windows physical-core discovery subprocess warnings.

## Verification

- Backend test suite: 31 passed.
- Accuracy Lab focused tests: 5 passed.
- API smoke checks: `/api/health`, `/api/datasets`, `/api/overview`, `/api/accuracy-lab/runs` all returned HTTP 200.
- End-to-end Accuracy Lab worker run on the included NSL-KDD random-holdout dataset completed and produced all three approach metrics and efficiency measurements.
- Frontend TSX syntax transpilation passed for `App.tsx` and `main.tsx`.
- Full frontend production build was not run because npm dependency installation timed out in this environment.

## Research integrity note

The NSL-KDD end-to-end smoke run used the random-holdout-from-training-file protocol. Its metrics are not official benchmark results and should not be presented as such. The smoke run was removed from the packaged experiment history after verification.
