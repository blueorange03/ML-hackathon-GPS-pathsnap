"""Train >=5 models, GroupKFold CV on train trips, evaluate on student_test, log everything."""
import pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from common import *
from telemetry import Telemetry
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


def xgb_device():
    """Use the GPU if this xgboost build + CUDA works, otherwise CPU."""
    try:
        XGBClassifier(device="cuda", n_estimators=2, tree_method="hist").fit(np.random.rand(20, 3), np.arange(20) % 2)
        return "cuda"
    except Exception as e:
        print("XGBoost GPU unavailable, using CPU:", str(e)[:80]); return "cpu"

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def pick_device():
    """Use the GPU for XGBoost if it works, otherwise fall back to CPU."""
    if xgb is None:
        return None
    try:
        xgb.XGBClassifier(n_estimators=2, device="cuda", tree_method="hist").fit(np.random.rand(20, 3), np.arange(20) % 2)
        return "cuda"
    except Exception:
        return "cpu"


DEV = pick_device()
print("XGBoost device:", DEV)

out = Path("outputs"); (out / "models").mkdir(parents=True, exist_ok=True)
C = pickle.load(open(out / "features.pkl", "rb"))
tr, te, net = C["train"], C["student_test"], C["net"]
Xtr, ytr, gtr = tr[FEATURES].values, tr.label.values, tr.trip.values
Xte, yte = te[FEATURES].values, te.label.values

MODELS = {   # name: (family, key config string, factory)
    "LogisticRegression": ("Linear", "C=1, scaled", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))),
    "GaussianNB": ("Probabilistic", "default var_smoothing", lambda: GaussianNB()),
    "RandomForest": ("Ensemble (bagging)", "100 trees, min_leaf=10, max_samples=0.3", lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_samples=0.3, n_jobs=-1, random_state=SEED)),
    "HistGradientBoosting": ("Boosting", "lr=0.1, 200 iter", lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, random_state=SEED)),
    **({"XGBoost": ("Boosting (GPU)" if DEV == "cuda" else "Boosting", f"300 trees, depth 6, lr=0.1, device={DEV}",
                    lambda: xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                                              device=DEV, eval_metric="logloss", random_state=SEED))} if DEV else {}),
    "MLP": ("Neural network", "(64,32), relu, early stop", lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), early_stopping=True, max_iter=200, random_state=SEED))),
}


if HAVE_XGB:
    DEV = xgb_device(); print("XGBoost device:", DEV)
    MODELS["XGBoost"] = ("Boosting (GPU)" if DEV == "cuda" else "Boosting", f"depth=6, lr=0.1, 300 trees, device={DEV}",
                         lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                                               device=DEV, random_state=SEED, eval_metric="logloss"))


ORDER = ["LogisticRegression", "GaussianNB", "XGBoost", "HistGradientBoosting", "MLP", "RandomForest"]
MODELS = {k: MODELS[k] for k in ORDER if k in MODELS}


def evaluate(model, df, X, y):
    t0 = time.perf_counter(); p = model.predict_proba(X)[:, 1]; inf = time.perf_counter() - t0
    top1, miss = top1_and_miss(df, p)
    pred = predicted_arcs(df, p)
    return dict(macro_f1=f1_score(y, (p >= .5).astype(int), average="macro"), top1=top1, miss_rate=miss,
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), continuity=route_continuity(pred, net),
                inf_ms_per_1k=1000 * inf / len(X) * 1000), p


fold_rows, summ = [], []
tel = Telemetry(); _total = len(MODELS) * 6; _step = 0; tel.begin('Train & compare', _total)
pbar = tqdm(total=_total, desc='models x folds')
def _live():
    keys = ['model', 'cv_macro_f1', 'test_macro_f1', 'test_top1', 'test_continuity', 'train_time_s']
    return [{k: float(r[k]) if k != 'model' else r[k] for k in keys} for r in summ]

for name, (fam, cfg, make) in MODELS.items():
    print("==", name)
    pbar.set_description(f"NOW: {name}")
    tel.update('Train & compare', _step, f'RUNNING NOW: {name}', _live())
    for k, (a, b) in enumerate(GroupKFold(5).split(Xtr, ytr, gtr)):   # group = trip
        m = make().fit(Xtr[a], ytr[a]); r, _ = evaluate(m, tr.iloc[b], Xtr[b], ytr[b])
        fold_rows.append(dict(model=name, fold=k, **r))
        _step += 1; pbar.update(1); tel.update('Train & compare', _step, f'{name}: CV fold {k+1}/5', _live())
    t0 = time.perf_counter(); m = make().fit(Xtr, ytr); train_s = time.perf_counter() - t0
    r, _ = evaluate(m, te, Xte, yte)
    cv = pd.DataFrame([f for f in fold_rows if f["model"] == name])
    summ.append(dict(model=name, family=fam, config=cfg, cv_macro_f1=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(),
                     cv_top1=cv.top1.mean(), cv_continuity=cv.continuity.mean(), cv_logloss=cv.log_loss.mean(),
                     test_macro_f1=r["macro_f1"], test_top1=r["top1"], test_miss_rate=r["miss_rate"], test_logloss=r["log_loss"],
                     test_continuity=r["continuity"], inf_ms_per_1k_cand=r["inf_ms_per_1k"], train_time_s=train_s, seed=SEED))
    _step += 1; pbar.update(1); pbar.set_postfix_str(f"{name} test F1={summ[-1]['test_macro_f1']:.3f}")
    tel.update('Train & compare', _step, f'{name}: final fit + student-test done', _live())
    pd.DataFrame(summ).to_csv(out / "model_comparison_partial.csv", index=False)
    joblib.dump(m, out / "models" / f"{name}.joblib")
    if name == "MLP":
        json.dump(m[-1].loss_curve_, open(out / "mlp_loss_curve.json", "w"))

tel.finish('Train & compare', 'all models trained')
pd.DataFrame(fold_rows).to_csv(out / "cv_fold_results.csv", index=False)
S = pd.DataFrame(summ); S.to_csv(out / "model_comparison.csv", index=False)
json.dump(dict(seed=SEED, features=FEATURES, models={k: v[1] for k, v in MODELS.items()}, cv="GroupKFold(5) by trip"),
          open(out / "experiment_config.json", "w"), indent=2)

for cols, fn, title in [(["cv_macro_f1"], "cv_f1.png", "CV Macro F1 (train, GroupKFold)"),
                        (["test_top1", "test_macro_f1"], "test_top1_f1.png", "Student-test Top-1 & Macro F1"),
                        (["test_logloss"], "logloss.png", "Student-test Log Loss")]:
    S.set_index("model")[cols].plot.bar(figsize=(8, 4), title=title); plt.xticks(rotation=25); plt.tight_layout()
    plt.savefig(out / fn, dpi=130); plt.close()
if (out / "mlp_loss_curve.json").exists():
    plt.plot(json.load(open(out / "mlp_loss_curve.json"))); plt.title("MLP training loss"); plt.xlabel("epoch")
    plt.savefig(out / "mlp_loss.png", dpi=130); plt.close()
print(S[["model", "cv_macro_f1", "test_macro_f1", "test_top1", "test_logloss", "test_continuity"]].round(4).to_string())
