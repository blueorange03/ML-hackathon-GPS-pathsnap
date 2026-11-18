# PathSnap – 11:30 checkpoint kit (preprocessing, features, model comparison)

Run from the repo root (put `Student folder/` next to `src/`, or pass its path):

```
pip install -r requirements.txt
cd src
python eda.py        "../Student folder"     # inventory, gaps, segments, plots, Folium map -> outputs/eda/
python make_features.py "../Student folder"  # candidates + segmented features -> outputs/features.pkl
python train_compare.py                       # 5 models, GroupKFold CV, student-test -> outputs/
python train_xgb_optuna.py                    # candidate-classification XGBoost tuning
python train_xgb_rank_optuna.py               # pairwise XGBoost ranking tuning; 100 TPE trials by default
python run_xgb_pipeline.py                    # EDA + features + XGBoost optimization in one command
```
The evaluator partition is never read. Everything uses seed 42 and is logged to CSV/JSON.

## 11:30 checklist (maps to the rubric: Preprocessing+FE = 5, Model selection+training = 5)
- [ ] Run `eda.py`; open `dataset_inventory.csv` and read the anomaly table. Note what you find (gaps, NaNs, bad route ids, disconnected networks).
- [ ] Check the Folium map: do the GPS points sit near the green route? If the route looks shifted, check coordinate order.
- [ ] Run `make_features.py`; note the **candidate miss rate** and positive rate printed/stored.
- [ ] Run `train_compare.py`; open `model_comparison.csv` and the PNGs.
- [ ] Write 3-4 sentences each: why you chose the features, what the models show, which is selected and why
      (primary metric: say whether it's CV macro F1, Top-1, or route continuity, and why).
- [ ] Be ready to explain every line in viva.

## Design decisions to be able to explain
- **Projection**: local equirectangular (metres) around the network centre; degrees are not distances.
- **Candidates**: for each GPS point, up to 10 nearest arcs within 150 m (nearest always kept), via exact point-to-segment distance.
- **Label**: candidate arc is in the trip's `.route` set. The route is a sequence without per-point alignment,
  so this is a proxy label (a route that revisits a road, or a parallel road, can blur it). State this limitation.
- **Top-1 accuracy / miss rate**: correct if the top-scored candidate is in the route set; miss if no candidate is.
- **Route continuity**: share of consecutive distinct predicted arcs where to_node(prev) == from_node(next).
- **CV**: `GroupKFold` by trip, so no trip is split across folds. Scalers sit inside pipelines (fit on train folds only).
- **Macro F1** uses a 0.5 threshold on candidate-level binary labels; positives are the minority class.
- Loss curve is saved only for the MLP (the only model here with an epoch history).

## Timestamp gaps and XGBoost optimization
- Both EDA and feature generation split observations when consecutive timestamps differ by more than 120 seconds by default. Override with a second CLI argument, for example `python make_features.py "../Student folder" 60`.
- Every observed fix remains in its original trip and keeps its observation index. No interpolation occurs. Previous/next-based features are clipped to each segment, and `dt_prev` is zero at each segment start.
- Feature rows include a segment ID and cache metadata contains each segment's original observation ranges and lengths. CV groups by original trip, keeping all segments from a trip in one fold; route continuity is computed within segments.
- `train_xgb_optuna.py` is the original candidate-classification tuning path. `train_xgb_rank_optuna.py` adds an XGBoost pairwise-ranking approach: candidates for each GPS observation form one ranking query, and Optuna maximizes pooled out-of-fold Top-1 under five-fold GroupKFold by original trip. It compares the ranker with the original XGBoost classifier baseline and selects using CV only; student-test labels are used only for the final development report. Ranking scores are not probabilities, so Macro F1 and Log Loss are reported only for the classifier. `PATHSNAP_OPTUNA_TRIALS` controls the ranker trial count (100 by default).
- HMM/Viterbi is not implemented in this repository.

### Simple run
Install dependencies once, then run the full pipeline:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_xgb_pipeline.py
```

To try more configurations, use `PATHSNAP_OPTUNA_TRIALS=100 python run_xgb_pipeline.py`. To use another data directory or threshold: `python run_xgb_pipeline.py "/path/to/Student folder" 120`. This does not rerun six-model selection. A 90% or 95% Top-1 score is a goal, not a guaranteed result; use the grouped-CV and student-test metrics in the generated comparison CSV to report what the run actually achieves.
