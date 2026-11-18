"""Build candidate/feature tables for train + student_test. Evaluator data is never read."""
import sys, pickle, time
from pathlib import Path
from joblib import Parallel, delayed
from common import *
from telemetry import Telemetry
from tqdm import tqdm


def process(split_dir, tid, gap_threshold_s):
    t0 = time.time()
    trip = load_trip(split_dir, tid)
    df = build_candidates(trip, gap_threshold_s=gap_threshold_s)
    seg_id, seg_ranges = segment_track(trip["track"], gap_threshold_s)
    print(f"  {tid}: {len(trip['track'])} obs, {len(trip['arcs'])} arcs -> "
          f"{0 if df is None else len(df)} candidates, {len(seg_ranges)} segments ({time.time()-t0:.1f}s)", flush=True)
    segment_meta = {"segment_ids": seg_id, "ranges": seg_ranges,
                    "lengths": [b - a for a, b in seg_ranges]}
    return tid, df, trip["arcs"], trip["route"], segment_meta


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "Student folder")
    gap_threshold_s = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GAP_THRESHOLD_S
    out = Path("outputs"); out.mkdir(exist_ok=True)
    tel = Telemetry()
    cache = {"net": {}, "route": {}, "segments": {}, "gap_threshold_s": gap_threshold_s}
    for split in ["train", "student_test"]:
        t0 = time.time(); print(f"[{split}]", flush=True)
        tids = list_trips(root / split); stage = f"Features: {split}"
        tel.begin(stage, len(tids)); res = []
        gen = Parallel(n_jobs=-1, return_as="generator_unordered")(delayed(process)(root / split, tid, gap_threshold_s) for tid in tids)
        for r in tqdm(gen, total=len(tids), desc=stage):
            res.append(r); tel.update(stage, len(res), f"last trip {r[0]}")
        tel.finish(stage, f"{len(res)} trips processed")
        frames = []
        for tid, df, arcs, route, segment_meta in res:
            if df is None:
                print(f"[skip] {split}/{tid}: empty or invalid files"); continue
            frames.append(df); cache["net"][tid] = arcs; cache["route"][tid] = route
            cache["segments"][tid] = segment_meta
        d = cache[split] = pd.concat(frames, ignore_index=True)
        print(f"{split}: {d.trip.nunique()} trips, {d.groupby(['trip','obs']).ngroups} observations, "
              f"{len(d)} candidates, positive rate {d.label.mean():.3f}, {time.time()-t0:.1f}s", flush=True)
    pickle.dump(cache, open(out / "features.pkl", "wb"))
