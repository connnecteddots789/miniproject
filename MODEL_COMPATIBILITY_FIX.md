# SENTINEL model compatibility fix

The bundled production model artifacts were serialized with scikit-learn 1.8.0.
The backend dependency pin is therefore 1.8.0 as well. Do not downgrade scikit-learn
to 1.6.1 while using these pre-trained artifacts; scikit-learn does not guarantee
pickle compatibility across versions.

On Windows, the backend sets `LOKY_MAX_CPU_COUNT` before importing the ML stack
to avoid loky's optional physical-core probe (which may invoke a missing system
utility on some Windows installations). This is only a worker-count setting and
does not change model predictions.

To install the locked backend dependencies from the SENTINEL project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

Restart the backend after dependency installation. Existing model artifacts are
retained; no retraining is required solely for this dependency alignment.
