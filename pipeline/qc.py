"""QC: per-clip metrics, qc_report.html (all clips) and a contact_sheet.mp4 per clip
(source video | front + side skeleton)."""
from __future__ import annotations
import html, subprocess
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from . import skeleton as S
from .common import read_json, write_json

BONES = [(S.IDX[j], S.IDX[S.PARENT[j]]) for j in S.JOINTS if S.PARENT[j]]


def jitter_mm(P):
    """Median-over-joints RMS deviation from a cubic Savitzky-Golay fit (high-frequency noise)."""
    from scipy.signal import savgol_filter
    if len(P) < 11:
        return 0.0
    sm = savgol_filter(P, 11, 3, axis=0, mode="interp")
    return float(np.median(np.sqrt(np.mean(np.sum((P - sm) ** 2, -1), 0))) * 1000)


def metrics(P, conf_kp, reproj, single_view, qc, cfg):
    q = dict(qc)
    q["jitter_mm"] = jitter_mm(P)
    body = conf_kp[:, :23] if conf_kp is not None else None
    q["low_conf_pct"] = float(np.mean(body < 0.3) * 100) if body is not None else None
    q["reproj_px"] = float(np.nanmedian(reproj)) if reproj is not None and np.isfinite(reproj).any() else None
    q["single_view"] = bool(single_view)
    t = cfg["qc"]
    checks = [q["foot_slide_cm"] <= t["max_foot_slide_cm"], q["jitter_mm"] <= t["max_jitter_mm"],
              (q["low_conf_pct"] or 0) <= t["max_low_conf_pct"],
              (q["reproj_px"] or 0) <= t["max_reproj_px"]]
    q["grade"] = "green" if all(checks) and not single_view else ("amber" if sum(checks) >= 3 else "red")
    return q


def _draw_skel(canvas, P2, color=(90, 230, 255)):
    for a, b in BONES:
        pa, pb = P2[a], P2[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(canvas, tuple(int(v) for v in pa), tuple(int(v) for v in pb), color, 2, cv2.LINE_AA)


def contact_sheet(video: Path | None, src_index, P, out: Path, fps=60, h=540, hits=(), contacts=None,
                  title=""):
    """Side-by-side preview. P (T,J,3) world positions (Y up)."""
    cap = cv2.VideoCapture(str(video)) if video else None
    vw = int(h * 9 / 16) if cap is None else None
    T = len(P)
    ctr = np.nanmean(P[:, S.IDX["Hips"]], 0)
    ext = max(1.0, float(np.nanmax(np.abs(P[..., [0, 2]] - ctr[[0, 2]]))) + 0.3)
    scale = h / max(2.2, ext * 2, np.nanmax(P[..., 1]) + 0.3)
    panel = int(h * 0.75)
    tmp = out.with_suffix(".raw.mp4")
    wr = None
    last_src = -1
    frame_src = None
    for t in range(T):
        if cap is not None:
            s = int(src_index[t]) if src_index is not None else t
            if s != last_src + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            ok, img = cap.read()
            last_src = s
            if ok:
                frame_src = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        left = frame_src if frame_src is not None else np.zeros((h, vw or panel, 3), np.uint8)
        views = []
        for axes, label in (((0, 1), "FRONT"), ((2, 1), "SIDE")):
            c = np.full((h, panel, 3), 18, np.uint8)
            cv2.line(c, (0, int(h * 0.92)), (panel, int(h * 0.92)), (60, 60, 60), 1)
            P2 = np.stack([(P[t, :, axes[0]] - ctr[axes[0]]) * scale * (1 if label == "SIDE" else -1) + panel / 2,
                           h * 0.92 - P[t, :, axes[1]] * scale], -1)
            _draw_skel(c, P2)
            if contacts:
                for side, col in (("Left", (80, 255, 120)), ("Right", (80, 160, 255))):
                    if contacts[side][t]:
                        p = P2[S.IDX[f"{side}ToeBase"]]
                        cv2.circle(c, (int(p[0]), int(p[1])), 6, col, -1, cv2.LINE_AA)
            cv2.putText(c, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
            views.append(c)
        frame = np.hstack([left] + views)
        cv2.putText(frame, f"{title}  {t:04d}", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)
        if t in hits:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, h - 1), (60, 60, 255), 6)
        if wr is None:
            wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], h))
        wr.write(frame)
    wr.release()
    if cap is not None:
        cap.release()
    # browser-friendly H.264
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-preset", "veryfast", "-crf", "24", "-movflags", "+faststart", str(out)])
    if r.returncode == 0:
        tmp.unlink()
    else:
        tmp.replace(out)


def report(out_root: Path):
    """Aggregate every *.clip.json under out/ into out/qc_report.html."""
    rows = []
    for p in sorted(out_root.rglob("*.clip.json")):
        m = read_json(p)
        q = m.get("qc", {})
        rel = p.parent.relative_to(out_root)
        rows.append((rel, m, q))
    col = {"green": "#2fd27a", "amber": "#ffb547", "red": "#ff5c5c"}
    fmt = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"

    def sync(q):
        c = q.get("chirp_check_ms")
        return q.get("sync_method", "-") + ("" if c is None else f" (chirp {c:+.1f} ms)")
    trs = "".join(
        f"<tr><td><span class=dot style='background:{col.get(q.get('grade'), '#888')}'></span>"
        f"{html.escape(m['take'])}</td><td>{m['frames']}</td><td>{fmt(q.get('foot_slide_cm'))}</td>"
        f"<td>{fmt(q.get('jitter_mm'))}</td><td>{fmt(q.get('low_conf_pct'))}</td><td>{fmt(q.get('reproj_px'))}</td>"
        f"<td>{sync(q)}</td>"
        f"<td>{'SINGLE VIEW' if q.get('single_view') else ''}</td>"
        f"<td><a href='{rel}/{m['take']}.contact.mp4'>preview</a></td></tr>" for rel, m, q in rows)
    doc = f"""<!doctype html><meta charset=utf-8><title>SLU Mocap QC</title>
<style>body{{font:14px -apple-system,Inter,sans-serif;background:#0d0f12;color:#e8e8e8;margin:32px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:8px 12px;border-bottom:1px solid #222;text-align:left}}
th{{color:#888;font-weight:500}}a{{color:#7cc4ff}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:10px}}</style>
<h2>SLU Mocap - QC</h2><table><tr><th>Take</th><th>Frames</th><th>Foot slide cm</th><th>Jitter mm</th>
<th>Low-conf %</th><th>Reproj px</th><th>Sync</th><th></th><th></th></tr>{trs}</table>"""
    (out_root / "qc_report.html").write_text(doc)
