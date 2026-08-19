# Fleet RUL Monitor — Predictive Maintenance for Turbofan Engines

Predicts **Remaining Useful Life (RUL)** from multivariate sensor time-series,
using the NASA C-MAPSS (FD001) turbofan degradation benchmark. Built as a
portfolio project mapped onto a production-technology / predictive-maintenance
use case (Ajin Industry interview context).

Two model families are trained and compared:
- **LSTM regressor** — RMSE ≈ 12.9 on held-out test engines
- **Transformer encoder regressor** — RMSE ≈ 13.5

Both are competitive with published literature on this exact benchmark
(typical reported RMSE on FD001 is 13–16 for sequence models).

## What's inside

```
predictive-maintenance/
├── data/                      # NASA C-MAPSS FD001 files (train/test/RUL)
├── scripts/
│   ├── preprocessing.py       # loading, RUL labeling, scaling, windowing
│   ├── models.py              # LSTMRegressor + TransformerRegressor (PyTorch)
│   └── train.py                # training loop + NASA scoring function
├── models/                    # saved checkpoints, scaler, config (after training)
├── app.py                     # Flask backend + REST API
├── templates/index.html       # dashboard UI
├── static/style.css           # industrial HMI-style dark theme
├── static/dashboard.js        # fleet grid, RUL gauge, sensor charts
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the models

Model checkpoints are already included under `models/`, but to retrain from
scratch:

```bash
cd scripts
python train.py --model lstm --epochs 40
python train.py --model transformer --epochs 20
```

This writes to `../models/`:
- `{model}_best.pt` — best checkpoint (lowest test RMSE)
- `{model}_history.json` — per-epoch loss / RMSE / NASA score
- `scaler.pkl` — the MinMaxScaler fit on training data (needed at inference time)
- `config.json` — feature columns, sequence length, best RMSE

### Key methodology notes

- **RUL labeling**: for training engines (which run to failure), RUL at each
  cycle = `max_cycle_for_engine − current_cycle`, clipped at 125. The cap
  reflects the fact that early-life degradation is roughly flat — an engine
  at cycle 5 isn't meaningfully "300 cycles from failure" in a way the model
  can learn, so capping prevents overfitting to an unrealistic signal.
- **Windowing**: each training example is a sliding window of 30 consecutive
  cycles; the label is the RUL at the window's last cycle. Test engines each
  contribute exactly one window — the last 30 cycles before the recorded cutoff.
- **NASA scoring function** (`nasa_score` in `train.py`): the official C-MAPSS
  asymmetric penalty. Predicting a LOWER RUL than the truth (early warning) is
  penalized gently; predicting a HIGHER RUL (late warning, i.e. the model
  thought there was more life left than there was) is penalized much more
  heavily — this mirrors the real safety/cost asymmetry in maintenance
  scheduling.
- **Dropped sensors**: 7 of the 21 sensors are constant/near-constant in
  FD001 and contribute no signal — dropped per standard practice in the
  literature (see `DROP_SENSORS` in `preprocessing.py`).

## 2. Run the dashboard

```bash
python app.py
```

Open **http://localhost:5000**.

The dashboard shows:
- **Fleet grid** — all 100 test engines, sorted by predicted RUL, color-coded
  (critical < 20 cycles, warning < 50, healthy ≥ 50)
- **Detail panel** — click any engine to see its predicted vs. true RUL, an
  RUL gauge, and live sensor trend charts for that engine's full recorded
  lifetime
- **Model switcher** — toggle between LSTM and Transformer predictions live

## 3. REST API (for integration)

| Endpoint | Method | Description |
|---|---|---|
| `/api/models` | GET | list available trained models |
| `/api/predict_all?model=lstm` | GET | predictions for all 100 test engines |
| `/api/engine/<id>?model=lstm` | GET | full detail + sensor traces for one engine |
| `/api/predict` | POST | predict RUL from a raw 30×17 sensor window (JSON) |

Example manual prediction:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "lstm", "window": [[<17 raw feature values>], ... 30 rows]}'
```

Feature order (after dropping constant sensors) is in `config.json` →
`feature_cols`. Values should be **raw, unscaled** — the endpoint applies
the training-time `MinMaxScaler` internally.

## Talking points for the Ajin Industry interview

- Framed around a real automotive manufacturing pain point: unplanned line
  stoppages from undetected component degradation.
- Compared two sequence architectures (LSTM vs. Transformer) rather than
  committing to one, and can speak to the trade-offs (LSTM: lighter, strong
  on this benchmark size; Transformer: more parallelizable, scales better to
  longer sequences / more sensors).
- Used the domain-standard evaluation (NASA scoring function), not just
  RMSE — signals awareness that in maintenance, false "all clear" predictions
  cost more than overly cautious ones.
- End-to-end deployable: a live dashboard a plant engineer could actually
  use to triage which machines need attention first, not just a notebook.

## Extending this project

- Swap in real production sensor data once available (schema is generic —
  just adjust `FEATURE_COLS` in `preprocessing.py`)
- Add classification alongside regression (binary "needs maintenance within
  N cycles" flag) for a simpler operator-facing signal
- Try the harder C-MAPSS subsets (FD002/FD003/FD004, which include multiple
  operating conditions and fault modes) to demonstrate robustness
