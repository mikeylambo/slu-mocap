"""triangulate: confidence-weighted DLT + RANSAC view rejection + bone-length gap filling,
then world alignment (Y up from the floor plane, facing +Z, metres).

Single-view fallback: Apple Vision 3D (saved by SLU Capture) or a geometric lifter
(Taylor-style orthographic depth from anthropometric bone lengths, temporally
disambiguated). Single-view takes are flagged in QC.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from . import skeleton as S
from .common import log


def projection(cam) -> np.ndarray:
    return cam["K"] @ np.hstack([cam["R"], cam["t"].reshape(3, 1)])


def undistort(pts, cam):
    """pixel -> undistorted pixel (keeps K) so a plain pinhole P can be used."""
    sh = pts.shape
    p = pts.reshape(-1, 1, 2).astype(np.float64)
    ok = np.isfinite(p).all(-1).ravel()
    out = np.full_like(p, np.nan)
    if ok.any():
        out[ok] = cv2.undistortPoints(p[ok], cam["K"], cam["dist"], P=cam["K"])
    return out.reshape(sh)


def dlt(Ps, pts, w):
    """Ps (V,3,4), pts (N,V,2), w (N,V) -> X (N,3). Weighted homogeneous DLT, batched."""
    N, V = w.shape
    A = np.zeros((N, 2 * V, 4))
    for v in range(V):
        x, y = pts[:, v, 0:1], pts[:, v, 1:2]
        A[:, 2 * v] = (x * Ps[v][2] - Ps[v][0]) * w[:, v:v + 1]
        A[:, 2 * v + 1] = (y * Ps[v][2] - Ps[v][1]) * w[:, v:v + 1]
    A = np.nan_to_num(A)
    # normalise rows for conditioning
    A /= np.linalg.norm(A, axis=2, keepdims=True) + 1e-12
    A *= np.repeat(w, 2, axis=1)[..., None]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1]
    return X[:, :3] / X[:, 3:4]


def reproject(Ps, X):
    Xh = np.concatenate([X, np.ones((*X.shape[:-1], 1))], -1)
    x = np.einsum("vij,nj->nvi", Ps, Xh)
    return x[..., :2] / x[..., 2:3]


def triangulate(cams: dict, kps: dict, conf_thr=0.25, reproj_thr_frac=0.012):
    """cams {role: cam}, kps {role: (P2 (T,133,2), C (T,133))} -> X (T,133,3), conf, err_px."""
    roles = list(kps)
    V = len(roles)
    T = kps[roles[0]][0].shape[0]
    Ps = np.stack([projection(cams[r]) for r in roles])
    pts = np.stack([undistort(kps[r][0], cams[r]) for r in roles], 2)     # T,133,V,2
    C = np.stack([kps[r][1] for r in roles], 2)                             # T,133,V
    diag = max(np.hypot(*cams[r]["size"]) for r in roles)
    thr = reproj_thr_frac * diag
    pts = pts.reshape(-1, V, 2)
    C = C.reshape(-1, V)
    w = np.where((C > conf_thr) & np.isfinite(pts).all(-1), C, 0.0)
    X = np.full((pts.shape[0], 3), np.nan)
    err = np.full(pts.shape[0], np.nan)
    active = (w > 0).sum(1) >= 2
    for _ in range(max(1, V - 1)):           # RANSAC-style: iteratively drop the worst view
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        Xi = dlt(Ps, pts[idx], w[idx])
        e = np.linalg.norm(reproject(Ps, Xi) - pts[idx], axis=-1)        # N,V
        e = np.where(w[idx] > 0, e, 0)
        worst = e.argmax(1)
        bad = e[np.arange(idx.size), worst] > thr
        X[idx] = Xi
        err[idx] = np.where(w[idx] > 0, e, np.nan).max(1) if V else 0
        # with 2 views a bad pair cannot be repaired -> invalid; with 3+ drop the worst view
        can_drop = (w[idx] > 0).sum(1) > 2
        w[idx[bad & can_drop], worst[bad & can_drop]] = 0
        kill = idx[bad & ~can_drop]
        X[kill] = np.nan
        active = np.zeros_like(active)
        active[idx[bad & can_drop]] = True
    # cheirality: must be in front of every camera that saw it
    for v, r in enumerate(roles):
        z = (X @ cams[r]["R"].T + cams[r]["t"])[:, 2]
        X[z <= 0] = np.nan
    conf = np.where(np.isfinite(X).all(-1), np.nanmean(np.where(w > 0, w, np.nan), 1), 0)
    conf = np.nan_to_num(conf) * np.exp(-np.nan_to_num(err) / (2 * thr))
    return X.reshape(T, -1, 3), conf.reshape(T, -1), err.reshape(T, -1)


def fill_gaps(X, conf, fps, max_gap_s=0.6):
    """PCHIP-interpolate short gaps; hold ends; then re-impose median bone lengths."""
    X = X.copy()
    T, K, _ = X.shape
    t = np.arange(T)
    filled = np.zeros((T, K), bool)
    for k in range(K):
        ok = np.isfinite(X[:, k]).all(-1)
        if ok.sum() < 2 or ok.all():
            continue
        f = PchipInterpolator(t[ok], X[ok, k], axis=0, extrapolate=False)
        miss = ~ok
        # gap lengths
        runs = np.diff(np.r_[0, miss.astype(int), 0])
        st, en = np.where(runs == 1)[0], np.where(runs == -1)[0]
        for a, b in zip(st, en):
            if a == 0 or b == T:
                X[a:b, k] = X[b if a == 0 else a - 1, k]
            elif (b - a) <= max_gap_s * fps:
                X[a:b, k] = f(t[a:b])
            else:
                X[a:b, k] = f(t[a:b])    # long gap: still interpolate, but flagged
            filled[a:b, k] = True
    # bone-length constraint on filled points (parent -> child order)
    for a, b in S.KP_BONES:
        L = np.nanmedian(np.linalg.norm(X[:, b] - X[:, a], axis=-1)[~filled[:, b] & ~filled[:, a]]) \
            if (~filled[:, b] & ~filled[:, a]).any() else np.nan
        if not np.isfinite(L):
            continue
        m = filled[:, b]
        d = X[m, b] - X[m, a]
        n = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
        X[m, b] = X[m, a] + d / n * L
    conf = np.where(filled, conf * 0.3, conf)
    return X, conf, filled


def world_align(X, conf):
    """Camera coords (y down) -> world: Y up from floor plane fit, origin under hips, facing +Z."""
    good = conf > 0.3
    up = np.array([0, -1.0, 0])   # points are in ref-camera coords (OpenCV), so -y is up-ish
    feet = [S.LANK, S.RANK, S.LHEEL, S.RHEEL, S.LBTOE, S.RBTOE, S.LSTOE, S.RSTOE]
    F = X[:, feet][good[:, feet]]
    F = F[np.isfinite(F).all(-1)]
    if len(F) > 50:
        h = F @ up
        low = F[h < np.percentile(h, 25)]
        n = _ransac_plane(low)
        if n is not None:
            up = n if n @ up > 0 else -n
    # rotation mapping 'up' -> +Y
    R1 = _rot_between(up, np.array([0, 1.0, 0]))
    Y = X @ R1.T
    # facing: left-hip -> right-hip cross up
    lr = Y[:, S.LHIP] - Y[:, S.RHIP]
    lr = np.nanmean(lr[: max(10, len(lr) // 5)], 0)
    fwd = np.cross(lr, [0, 1, 0])
    yaw = np.arctan2(fwd[0], fwd[2])
    c, s = np.cos(-yaw), np.sin(-yaw)
    R2 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Y = Y @ R2.T
    floor = np.nanpercentile(Y[:, feet, 1], 2)
    hips0 = np.nanmean(0.5 * (Y[: 10, S.LHIP] + Y[: 10, S.RHIP]), 0)
    Y = Y - np.array([hips0[0], floor, hips0[2]])
    return Y, (R2 @ R1)


def _rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), a @ b
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def _ransac_plane(P, iters=300, thr=0.02, rng=np.random.default_rng(0)):
    best, bn = 0, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < thr
        if inl.sum() > best:
            best, bn = inl.sum(), n
    return bn


# ---- single view -----------------------------------------------------------------------
VISION3D_MAP = {  # Apple VNHumanBodyPose3DObservation.JointName raw values -> COCO idx
    "left_shoulder_1_joint": S.LSHO, "right_shoulder_1_joint": S.RSHO,
    "left_forearm_joint": S.LELB, "right_forearm_joint": S.RELB,
    "left_hand_joint": S.LWRI, "right_hand_joint": S.RWRI,
    "left_upLeg_joint": S.LHIP, "right_upLeg_joint": S.RHIP,
    "left_leg_joint": S.LKNE, "right_leg_joint": S.RKNE,
    "left_foot_joint": S.LANK, "right_foot_joint": S.RANK,
    "head_joint": S.NOSE,
}


def from_vision3d(frames: list[dict], T: int, fps: float, start_s: float = 0.0):
    """Vision 3D json [{t, joints:{name:[x,y,z]}}] (camera space, m) -> X (T,133,3), conf."""
    X = np.full((T, S.N_KP, 3), np.nan)
    ts = np.array([f["t"] for f in frames]) - start_s
    for k_name, k in VISION3D_MAP.items():
        v = np.array([f["joints"].get(k_name, [np.nan] * 3) for f in frames], float)
        for d in range(3):
            X[:, k, d] = np.interp(np.arange(T) / fps, ts, v[:, d], left=np.nan, right=np.nan)
    # ears ~ head, toes/heels from ankle (flat feet assumption)
    for e in (S.LEAR, S.REAR, S.LEYE, S.REYE):
        X[:, e] = X[:, S.NOSE]
    for ank, toes, heel in ((S.LANK, (S.LBTOE, S.LSTOE), S.LHEEL), (S.RANK, (S.RBTOE, S.RSTOE), S.RHEEL)):
        for t_ in toes:
            X[:, t_] = X[:, ank] + np.array([0, 0.07, -0.15])   # camera coords: forward = -z towards cam
        X[:, heel] = X[:, ank] + np.array([0, 0.06, 0.05])
    conf = np.where(np.isfinite(X).all(-1), 0.5, 0)
    # Vision camera space is y-up; convert to OpenCV (y down, z forward)
    X = X * np.array([1, -1, -1])
    return X, conf


def lift_single_view(P2, C, cam, lengths=None):
    """Geometric 2D->3D lift (no learned prior => commercially clean).

    Scaled-orthographic model: per-frame scale from torso size, root depth from focal length,
    per-bone |dz| from known lengths, sign chosen for temporal continuity (then anatomical bias).
    """
    T = P2.shape[0]
    p = undistort(P2, cam)
    f = cam["K"][0, 0]
    c = cam["K"][:2, 2]
    torso2d = np.linalg.norm(0.5 * (p[:, S.LSHO] + p[:, S.RSHO]) - 0.5 * (p[:, S.LHIP] + p[:, S.RHIP]), axis=-1)
    torso3d = 0.50
    s = np.nan_to_num(_smooth1d(torso2d / torso3d, 15), nan=np.nanmedian(torso2d / torso3d))
    Z0 = f / np.maximum(s, 1e-6)                         # root depth (m)
    X = np.full((T, S.N_KP, 3), np.nan)
    X[..., :2] = (p - c) / s[:, None, None]
    X[..., 2] = 0.0
    lengths = lengths or {}
    order = [(a, b) for a, b in S.KP_BONES]
    prev = None
    for t in range(T):
        z = np.zeros(S.N_KP)
        for a, b in order:
            L = lengths.get((a, b))
            if L is None:
                L = np.nanmedian(np.linalg.norm(X[:, b, :2] - X[:, a, :2], axis=-1)) * 1.15
                lengths[(a, b)] = L
            dxy = np.linalg.norm(X[t, b, :2] - X[t, a, :2])
            dz = np.sqrt(max(L * L - dxy * dxy, 0.0)) * 0.8
            cand = (z[a] + dz, z[a] - dz)
            if prev is not None:
                z[b] = min(cand, key=lambda v: abs(v - prev[b]))
            else:
                z[b] = z[a] - dz if b in (S.LWRI, S.RWRI, S.LELB, S.RELB) else z[a]
        prev = z
        X[t, :, 2] = z
    X[..., 2] += Z0[:, None]
    X[..., :2] *= 1.0
    return X, C * 0.6


def _smooth1d(x, k):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return x
    x = np.interp(np.arange(len(x)), np.where(ok)[0], x[ok])
    ker = np.ones(k) / k
    return np.convolve(np.pad(x, k // 2, mode="edge"), ker, "valid")[: len(x)]
