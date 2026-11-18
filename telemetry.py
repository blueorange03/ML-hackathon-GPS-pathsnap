"""Live progress telemetry. Writes outputs/telemetry.json and outputs/dashboard.html
(auto-refreshes every 2 s). Open dashboard.html in a browser while scripts run."""
import json, time, html
from pathlib import Path

ORDER = ["EDA", "Features: train", "Features: student_test", "Train & compare"]


def _fmt(sec):
    if sec is None: return "-"
    sec = int(sec); m, s = divmod(sec, 60); h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


class Telemetry:
    def __init__(self, path="outputs/telemetry.json"):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        try: self.state = json.loads(self.path.read_text())
        except Exception: self.state = {"stages": {}, "results": []}

    def begin(self, stage, total):
        self.state["stages"][stage] = dict(start=time.time(), updated=time.time(), done=0, total=total,
                                           msg="starting", status="running")
        self._write()

    def update(self, stage, done, msg="", results=None):
        s = self.state["stages"][stage]
        s.update(done=done, msg=msg, updated=time.time(), status="running")
        if results is not None: self.state["results"] = results
        self._write()

    def finish(self, stage, msg="finished"):
        s = self.state["stages"][stage]
        s.update(done=s["total"], msg=msg, updated=time.time(), status="done"); self._write()

    def _write(self):
        try:
            self.path.write_text(json.dumps(self.state))
            (self.path.parent / "dashboard.html").write_text(self._html())
        except OSError:
            pass  # never let telemetry crash a run (e.g. file locked by the browser)

    def _html(self):
        rows = ""
        names = [n for n in ORDER if n in self.state["stages"]] + \
                [n for n in self.state["stages"] if n not in ORDER]
        for n in ORDER + [n for n in names if n not in ORDER]:
            s = self.state["stages"].get(n)
            if not s:
                rows += f"<tr><td>{n}</td><td colspan=4 class=g>not started</td></tr>"; continue
            pct = 100 * s["done"] / max(s["total"], 1)
            el = (s["updated"] if s["status"] == "done" else time.time()) - s["start"]
            eta = None if s["status"] == "done" or s["done"] == 0 else el / s["done"] * (s["total"] - s["done"])
            col = "#2e9e5b" if s["status"] == "done" else "#2f6fde"
            rows += (f"<tr><td><b>{html.escape(n)}</b></td>"
                     f"<td style='width:38%'><div class=bar><div style='width:{pct:.1f}%;background:{col}'></div></div></td>"
                     f"<td>{s['done']}/{s['total']} ({pct:.0f}%)</td><td>elapsed {_fmt(el)}<br>"
                     f"{'ETA ' + _fmt(eta) if eta is not None else ('done' if s['status']=='done' else 'ETA -')}</td>"
                     f"<td class=g>{html.escape(str(s['msg']))}</td></tr>")
        res = ""
        if self.state["results"]:
            cols = list(self.state["results"][0].keys())
            res = "<h3>Results so far (student-test)</h3><table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
            for r in self.state["results"]:
                res += "<tr>" + "".join(f"<td>{r[c]:.4f}</td>" if isinstance(r[c], float) else f"<td>{r[c]}</td>" for c in cols) + "</tr>"
            res += "</table>"
        return f"""<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=2>
<title>PathSnap telemetry</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
table{{border-collapse:collapse;width:100%;margin-bottom:20px}}td,th{{padding:8px 10px;border-bottom:1px solid #333;text-align:left}}
.bar{{background:#333;border-radius:6px;height:16px;overflow:hidden}}.bar div{{height:16px}}.g{{color:#999}}
</style></head><body><h2>PathSnap pipeline telemetry</h2><p class=g>Refreshes every 2 s. Last write: {time.strftime('%H:%M:%S')}</p>
<table><tr><th>Stage</th><th>Progress</th><th>Done</th><th>Time</th><th>Now</th></tr>{rows}</table>{res}</body></html>"""
