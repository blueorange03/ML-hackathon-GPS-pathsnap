# PathSnap – 11:30 checkpoint kit (preprocessing, features, model comparison)

Run from the repo root (put `Student folder/` next to `src/`, or pass its path):

```
pip install -r requirements.txt
cd src
python eda.py        "../Student folder"     # inventory, anomalies, plots, Folium map -> outputs/eda/
python make_features.py "../Student folder"  # candidates + features -> outputs/features.pkl
python train_compare.py                       # 5 models, GroupKFold CV, student-test -> outputs/
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

## Next (after 11:30): Optuna on the chosen model with GroupKFold, then HMM/Viterbi, FastAPI, UI.
