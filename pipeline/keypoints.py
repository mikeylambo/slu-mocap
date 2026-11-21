"""keypoints: RTMW whole-body (133 kps: body, feet, face, hands) via rtmlib (Apache-2.0).

Runs on GPU (onnxruntime-gpu) in Colab, CPU elsewhere. Tracks the main performer across
frames (largest person, then IoU continuity). Output cached per video:
    cache/<take>/<role>.kp.npz   kp (T,133,2) px, score (T,133), fps, size
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import log

_MODEL = None


def _model(device="cuda", mode="performance"):
    global _MODEL
    if _MODEL is None:
        from rtmlib import Wholebody
        try:
            _MODEL = Wholebody(mode=mode, backend="onnxruntime", device=device, to_openpose=False)
        except Exception:
            _MODEL = Wholebody(mode=mode, backend="onnxruntime", device="cpu", to_openpose=False)
    return _MODEL


def _bbox(kp, sc, thr=0.3):
    m = sc > thr
    if m.sum() < 4:
        return None
    p = kp[m]
    return np.array([*p.min(0), *p.max(0)])


def _iou(a, b):
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ar = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (ar(a) + ar(b) - inter + 1e-9)


def run_video(video: Path, out: Path, device="cuda", mode="performance") -> Path:
    import cv2
    if out.exists():
        return out
    model = _model(device, mode)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    KP, SC, prev = [], [], None
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        kps, scs = model(img)
        if kps is None or len(kps) == 0:
            KP.append(np.full((133, 2), np.nan)); SC.append(np.zeros(133)); i += 1
            continue
        boxes = [_bbox(k, s) for k, s in zip(kps, scs)]
        cands = [(j, b) for j, b in enumerate(boxes) if b is not None]
        if not cands:
            KP.append(np.full((133, 2), np.nan)); SC.append(np.zeros(133)); i += 1
            continue
        if prev is None:
            j = max(cands, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))[0]
        else:
            j = max(cands, key=lambda c: _iou(prev, c[1]) + 1e-6 * (c[1][3] - c[1][1]))[0]
        prev = boxes[j]
        KP.append(kps[j]); SC.append(scs[j])
        i += 1
        if i % 300 == 0:
            log("keypoints", f"{video.name}: {i}/{n}")
    cap.release()
    h, w = img.shape[:2] if KP else (0, 0)
    kp, sc = np.asarray(KP, np.float32), np.asarray(SC, np.float32)
    # rtmlib scores are SimCC logits-ish (~0..10); normalise to 0..1 confidence
    if sc.size and np.nanmax(sc) > 1.5:
        sc = np.clip(sc / 8.0, 0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, kp=kp, score=sc, fps=fps, size=np.array([w, h]))
    log("keypoints", f"{video.name}: {len(kp)} frames, mean conf {np.mean(sc):.2f}")
    return out


def load(path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def resample(kp: dict, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample 2D keypoints at arbitrary (sub-frame) times by linear interpolation."""
    fps = float(kp["fps"])
    f = times_s * fps
    T = kp["kp"].shape[0]
    i0 = np.clip(np.floor(f).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    a = np.clip(f - i0, 0, 1)[:, None]
    P = kp["kp"][i0] * (1 - a[..., None]) + kp["kp"][i1] * a[..., None]
    a1 = a[:, 0]
    S = np.where((a1 < 1e-3)[:, None], kp["score"][i0],
                 np.where((a1 > 1 - 1e-3)[:, None], kp["score"][i1], np.minimum(kp["score"][i0], kp["score"][i1])))
    out = (f < -0.5) | (f > T - 0.5)
    S[out] = 0
    return P, S
