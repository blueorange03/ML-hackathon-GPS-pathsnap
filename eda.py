"""EDA: inventory, missing values, time gaps, network connectivity, plots, Folium map."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, folium
from common import *
from telemetry import Telemetry
from tqdm import tqdm

root = Path(sys.argv[1] if len(sys.argv) > 1 else "Student folder")
gap_threshold_s = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GAP_THRESHOLD_S
out = Path("outputs/eda"); out.mkdir(parents=True, exist_ok=True)
rows, trips = [], {}
tel = Telemetry()
_total = sum(len(list_trips(root / sp)) for sp in ['train', 'student_test']); tel.begin('EDA', _total); _done = 0
for split in ["train", "student_test"]:           # evaluator partition is never touched
    for tid in list_trips(root / split):
        tr = load_trip(root / split, tid); trips[(split, tid)] = tr
        _done += 1; tel.update('EDA', _done, f'inspecting {split}/{tid}')
        if _done % 10 == 0: print(f'EDA {_done}/{_total}', flush=True)
        t, g, a, r = tr["track"], tr["track"], tr["arcs"], tr["route"]
        nan = int(np.isnan(t).sum()) if len(t) else 0
        lon_bad = int(((g[:, 0] < -180) | (g[:, 0] > 180)).sum()) if len(g) else 0
        lat_bad = int(((g[:, 1] < -90) | (g[:, 1] > 90)).sum()) if len(g) else 0
        dts = np.diff(t[:, 2]) if len(t) > 1 else np.array([])
        gap_ids = np.flatnonzero(dts > gap_threshold_s)
        seg_ids, seg_ranges = segment_track(t, gap_threshold_s)
        G = nx.Graph(); G.add_nodes_from(range(len(tr["nodes"])))
        G.add_edges_from(map(tuple, a)) if len(a) else None
        ncomp = nx.number_connected_components(G) if len(tr["nodes"]) else 0
        iso = int(sum(1 for _ in nx.isolates(G)))
        route_ok = (len(r) > 0 and r.max() < len(a))
        route_conn = float((a[r[:-1], 1] == a[r[1:], 0]).mean()) if route_ok and len(r) > 1 else np.nan
        P = Projector(tr["nodes"][:, 0].mean(), tr["nodes"][:, 1].mean())(g[:, 0], g[:, 1]) if len(g) and len(tr["nodes"]) else np.zeros((0, 2))
        rows.append(dict(split=split, trip=tid, n_obs=len(t), n_nodes=len(tr["nodes"]), n_arcs=len(a),
                         route_len=len(r), nan_cells=nan, lon_invalid=lon_bad, lat_invalid=lat_bad,
                         dup_ts=int((dts == 0).sum()), nonmonotonic_ts=int((dts < 0).sum()),
                         frac_dt_1s=float((dts == 1).mean()) if len(dts) else np.nan,
                         max_gap_s=float(dts.max()) if len(dts) else np.nan,
                         large_gap_count=len(gap_ids), segment_count=len(seg_ranges),
                         segment_lengths=";".join(str(b - a) for a, b in seg_ranges),
                         large_gap_seconds=";".join(f"{dts[i]:.3f}" for i in gap_ids),
                         components=ncomp, isolated_nodes=iso, route_ids_valid=route_ok,
                         route_arc_connectivity=route_conn,
                         path_len_m=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0,
                         lon_min=g[:, 0].min() if len(g) else np.nan, lon_max=g[:, 0].max() if len(g) else np.nan,
                         lat_min=g[:, 1].min() if len(g) else np.nan, lat_max=g[:, 1].max() if len(g) else np.nan))
inv = pd.DataFrame(rows); inv.to_csv(out / "dataset_inventory.csv", index=False)
gap_rows = []
for split in ["train", "student_test"]:
    for tid in list_trips(root / split):
        t = trips[(split, tid)]["track"]
        dts = np.diff(t[:, 2]) if len(t) > 1 else np.array([])
        for ix in np.flatnonzero(dts > gap_threshold_s):
            gap_rows.append(dict(split=split, trip=tid, obs_before=int(ix), obs_after=int(ix + 1),
                                 gap_s=float(dts[ix])))
pd.DataFrame(gap_rows, columns=["split", "trip", "obs_before", "obs_after", "gap_s"]).to_csv(
    out / "large_gaps.csv", index=False)
print(inv.groupby("split").agg(trips=("trip", "count"), obs_mean=("n_obs", "mean"), arcs_mean=("n_arcs", "mean"),
                               max_gap=("max_gap_s", "max"), nan=("nan_cells", "sum")).to_string())
print(f"Trips with gaps > {gap_threshold_s:g}s:")
print(inv[inv.large_gap_count > 0][["split", "trip", "large_gap_count", "segment_count",
                                    "segment_lengths", "large_gap_seconds"]].to_string(index=False))
print(f"Gap threshold: {gap_threshold_s:g}s; gap report: {out / 'large_gaps.csv'}")

# plots (partition = train)
tr_inv = inv[inv.split == "train"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(tr_inv.n_obs, 20); ax[0].set_title("GPS obs per trip (train)")
ax[1].hist(tr_inv.max_gap_s.dropna(), 20); ax[1].set_title("Max time gap per trip, s (train)")
ax[2].hist(tr_inv.n_arcs, 20); ax[2].set_title("Road arcs per trip network (train)")
plt.tight_layout(); plt.savefig(out / "distributions.png", dpi=130); plt.close()

# GPS trajectory + network + route plots for first 4 train trips
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, tid in zip(axes.ravel(), list_trips(root / "train")[:4]):
    tr = trips[("train", tid)]; N = tr["nodes"]; a = tr["arcs"]
    for f, t_ in a[::max(1, len(a) // 4000)]:
        ax.plot([N[f, 0], N[t_, 0]], [N[f, 1], N[t_, 1]], c="lightgrey", lw=.5)
    if len(tr["route"]) and tr["route"].max() < len(a):
        for ar in tr["route"]:
            f, t_ = a[ar]; ax.plot([N[f, 0], N[t_, 0]], [N[f, 1], N[t_, 1]], c="green", lw=1.5)
    ax.scatter(tr["track"][:, 0], tr["track"][:, 1], s=3, c="blue"); ax.set_title(f"train trip {tid}")
plt.tight_layout(); plt.savefig(out / "trajectories_network.png", dpi=130); plt.close()

# Folium map of one train trip: GPS, nearby network, reference route
tid = list_trips(root / "train")[0]; tr = trips[("train", tid)]
N, a, g = tr["nodes"], tr["arcs"], tr["track"]
m = folium.Map(location=[g[:, 1].mean(), g[:, 0].mean()], zoom_start=16, tiles=None)
fn = folium.FeatureGroup("Road network (near trip)"); fr = folium.FeatureGroup("Reference route")
fg = folium.FeatureGroup("GPS observations")
pad = 0.003
sel = (N[a[:, 0], 0] > g[:, 0].min() - pad) & (N[a[:, 0], 0] < g[:, 0].max() + pad) & \
      (N[a[:, 0], 1] > g[:, 1].min() - pad) & (N[a[:, 0], 1] < g[:, 1].max() + pad)
for f, t_ in a[sel]:
    folium.PolyLine([(N[f, 1], N[f, 0]), (N[t_, 1], N[t_, 0])], color="grey", weight=2).add_to(fn)
if len(tr["route"]) and tr["route"].max() < len(a):
    for ar in tr["route"]:
        f, t_ = a[ar]; folium.PolyLine([(N[f, 1], N[f, 0]), (N[t_, 1], N[t_, 0])], color="green", weight=5).add_to(fr)
for lon, lat, _ in g:
    folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True).add_to(fg)
for grp in (fn, fr, fg): grp.add_to(m)
folium.LayerControl().add_to(m)
m.save(out / f"folium_train_{tid}.html"); print("saved", out / f"folium_train_{tid}.html")

tel.finish("EDA", "inventory, plots and Folium map saved")
