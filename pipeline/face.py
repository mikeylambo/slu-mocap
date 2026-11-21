"""Face: ingest Live Link Face (ARKit) CSV exports -> 60 fps blendshape curves per take.

Name the Live Link Face slate with the take id shown on the Leader (e.g. kairos__taunt_a__t003)
and drop the CSV into inbox/face/. Output: out/<game>/<move>/<take>.face.json (52 ARKit curves).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .common import TAKE_RE, log, write_json


def _timecode_s(tc: str, fps_tc: float) -> float:
    h, m, s, f = tc.replace(";", ":").split(":")[:4]
    return int(h) * 3600 + int(m) * 60 + int(s) + float(f) / fps_tc


def convert(csv_path: Path, out_root: Path, fps=60.0):
    stem = csv_path.stem.split("_iPhone")[0]
    take = next((stem[:i] for i in range(len(stem), 0, -1) if TAKE_RE.match(stem[:i] + "__leader")), None)
    if not take:
        log("face", f"{csv_path.name}: slate is not a take id - skipped"); return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    keys = [k for k in rows[0] if k not in ("Timecode", "BlendShapeCount")]
    t = np.array([_timecode_s(r["Timecode"], 60.0) for r in rows])
    t -= t[0]
    V = np.array([[float(r[k]) for k in keys] for r in rows])
    tt = np.arange(0, t[-1], 1 / fps)
    curves = {k: np.round(np.interp(tt, t, V[:, i]), 4).tolist() for i, k in enumerate(keys)}
    g, m, _ = take.split("__")
    out = out_root / g / m / f"{take}.face.json"
    write_json(out, {"take": take, "fps": fps, "frames": len(tt), "curves": curves})
    log("face", f"{csv_path.name} -> {out.name} ({len(keys)} curves)")
    return out


def run(inbox: Path, out_root: Path):
    for p in sorted((inbox / "face").glob("*.csv")) if (inbox / "face").exists() else []:
        if not any(out_root.rglob(f"{p.stem.split('_iPhone')[0]}*.face.json")):
            convert(p, out_root)
