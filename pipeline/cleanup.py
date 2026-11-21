"""cleanup: One-Euro smoothing, foot contacts + foot-lock IK, trim, root motion, loop
crossfade, KAIROS stylize (anticipation->snap time remap, global speed), hit-frame events.

All operations work on a Clip dict: {fps, root (T,3), quats (T,J,4), lengths, offsets, ...}.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks
from . import skeleton as S, solve as SV

I = S.IDX
EXTREMITIES = {j for j in S.JOINTS if any(k in j for k in ("Hand", "Foot", "Toe", "ForeArm", "Leg", "Head"))
               and "UpLeg" not in j}


# ---- One-Euro filter ------------------------------------------------------------------------
def one_euro(x, fps, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
    """Vectorised One-Euro over axis 0. x (T,...) with NaNs held."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    alpha = lambda fc: 1.0 / (1.0 + fps / (2 * np.pi * fc))
    prev = x[0].copy()
    dprev = np.zeros_like(prev)
    out[0] = prev
    for t in range(1, len(x)):
        cur = np.where(np.isfinite(x[t]), x[t], prev)
        dx = (cur - prev) * fps
        ad = alpha(d_cutoff)
        dprev = ad * dx + (1 - ad) * dprev
        fc = min_cutoff + beta * np.abs(dprev)
        a = alpha(fc)
        prev = a * cur + (1 - a) * prev
        out[t] = prev
    return out


def smooth_positions(Pj, fps, cfg):
    """Zero-phase One-Euro (forward + backward pass) with torso vs extremity settings."""
    out = Pj.copy()
    for group, key in ((EXTREMITIES, "extremities"), (set(S.JOINTS) - EXTREMITIES, "torso")):
        idx = [I[j] for j in group]
        c = cfg["one_euro"][key]
        f = one_euro(Pj[:, idx], fps, c["min_cutoff"], c["beta"])
        b = one_euro(f[::-1], fps, c["min_cutoff"], c["beta"])[::-1]
        out[:, idx] = b
    return out


# ---- feet ---------------------------------------------------------------------------------------
def foot_contacts(P, fps, cfg):
    """P (T,J,3) world, Y up. Returns {side: (contact bool (T,), w_toe (T,))}."""
    out = {}
    c = cfg["contacts"]
    for s in ("Left", "Right"):
        ank, toe = P[:, I[f"{s}Foot"]], P[:, I[f"{s}ToeBase"]]
        ank_h0 = np.percentile(ank[:, 1], 5)
        toe_h0 = np.percentile(toe[:, 1], 5)
        h_ank, h_toe = ank[:, 1] - ank_h0, toe[:, 1] - toe_h0
        v = lambda p: np.linalg.norm(np.gradient(p[:, [0, 2]], axis=0), axis=-1) * fps
        v_ank, v_toe = uniform_filter1d(v(ank), 3), uniform_filter1d(v(toe), 3)
        heel_on = (h_ank < c["height_m"]) & (v_ank < c["speed_mps"])
        toe_on = (h_toe < c["height_m"]) & (v_toe < c["speed_mps"])
        con = median_filter((heel_on | toe_on).astype(np.uint8), 5).astype(bool)
        con = _drop_short(con, int(c["min_frames"]))
        w_toe = 1 / (1 + np.exp(-((h_ank - h_toe) - 0.0) / 0.01))
        w_toe = np.where(toe_on & ~heel_on, 1.0, np.where(heel_on & ~toe_on, 0.0, w_toe))
        out[s] = (con, uniform_filter1d(w_toe, 5))
    return out


def _drop_short(m, n):
    m = m.copy()
    d = np.diff(np.r_[0, m.astype(int), 0])
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if b - a < n:
            m[a:b] = False
    return m


def _runs(m):
    d = np.diff(np.r_[0, m.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def two_bone_ik(A, B, C, target, la, lb):
    """Move chain A-B-C so C reaches target, keeping lengths and the bend plane through B."""
    d = target - A
    dist = np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6, (la + lb) * 0.9999)
    dn = d / np.linalg.norm(d, axis=-1, keepdims=True)
    pole = B - A
    pole = pole - dn * np.sum(pole * dn, -1, keepdims=True)
    pole /= np.linalg.norm(pole, axis=-1, keepdims=True) + 1e-12
    cos_a = (la * la + dist * dist - lb * lb) / (2 * la * dist)
    sin_a = np.sqrt(np.clip(1 - cos_a * cos_a, 0, 1))
    Bn = A + la * (dn * cos_a + pole * sin_a)
    Cn = A + dn * dist
    return Bn, Cn


def foot_lock(P, contacts, L, fps, blend_frames=6):
    """Pin feet during contact (rolling heel->toe pivot), solve legs with 2-bone IK."""
    P = P.copy()
    for s, (con, w_toe) in contacts.items():
        f, t, te = I[f"{s}Foot"], I[f"{s}ToeBase"], I[f"{s}Toe_End"]
        delta = np.zeros((len(P), 3))
        wgt = np.zeros(len(P))
        for a, b in _runs(con):
            seg = slice(a, b)
            wt = w_toe[seg]
            heel, toe = P[seg, f], P[seg, t]
            A_toe = np.average(toe, 0, weights=wt + 1e-3)
            A_heel = np.average(heel, 0, weights=(1 - wt) + 1e-3)
            d = wt[:, None] * (A_toe - toe) + (1 - wt[:, None]) * (A_heel - heel)
            d[:, 1] = 0
            delta[seg] = d
            wgt[seg] = 1.0
            # blend out after the run so the release is smooth
            for i in range(1, blend_frames + 1):
                if b - 1 + i < len(P) and not con[b - 1 + i]:
                    delta[b - 1 + i] = d[-1]
                    wgt[b - 1 + i] = max(wgt[b - 1 + i], 1 - i / (blend_frames + 1))
                if a - i >= 0 and not con[a - i]:
                    delta[a - i] = d[0]
                    wgt[a - i] = max(wgt[a - i], 1 - i / (blend_frames + 1))
        shift = delta * wgt[:, None]
        # no floor penetration
        low = np.minimum(P[:, t, 1], P[:, te, 1])
        shift[:, 1] += np.maximum(0, -low)
        newF = P[:, f] + shift
        hip, knee = P[:, I[f"{s}UpLeg"]], P[:, I[f"{s}Leg"]]
        Bn, Cn = two_bone_ik(hip, knee, P[:, f], newF, L[f"{s}Leg"], L[f"{s}Foot"])
        rigid = Cn - P[:, f]
        P[:, I[f"{s}Leg"]] = Bn
        for j in (f, t, te):
            P[:, j] = P[:, j] + rigid
    return P


def foot_slide_cm(P, contacts, fps):
    """Mean / max horizontal drift (cm) of the planted point per contact."""
    vals = []
    for s, (con, w_toe) in contacts.items():
        for a, b in _runs(con):
            if b - a < 3:
                continue
            on_toe = w_toe[a:b] > 0.5
            worst = 0.0
            for m, j in ((on_toe, "ToeBase"), (~on_toe, "Foot")):
                if m.sum() >= 3:
                    pts = P[a:b][m, I[f"{s}{j}"]][:, [0, 2]]
                    d = np.linalg.norm(pts - np.median(pts, 0), axis=-1)
                    worst = max(worst, np.percentile(d, 90) * 100)
            vals.append(worst)
    return (float(np.mean(vals)), float(np.max(vals))) if vals else (0.0, 0.0)


# ---- time ---------------------------------------------------------------------------------------
def motion_energy(P, fps):
    idx = [I[j] for j in ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head", "Hips")]
    v = np.linalg.norm(np.gradient(P[:, idx], axis=0), axis=-1) * fps
    return uniform_filter1d(v.max(1), 7)


def trim(clip, P, cfg):
    fps = clip["fps"]
    T = len(P)
    a = 0
    tp = clip.get("tpose_frames")
    if tp is not None and len(tp) and tp[0] < 0.25 * T:
        a = int(tp[-1]) + 1 if tp[-1] - tp[0] < 0.5 * T else 0
    e = motion_energy(P, fps)
    moving = np.where(e > cfg["trim"]["still_mps"])[0]
    pad = int(cfg["trim"]["pad_s"] * fps)
    if len(moving):
        a = max(a, moving[moving >= a][0] - pad) if (moving >= a).any() else a
        b = min(T, moving[-1] + pad + 1)
    else:
        b = T
    if b - a < 0.3 * fps:
        return slice(0, T)
    return slice(max(0, a), b)


def nlerp(q0, q1, w):
    q1 = np.where(np.sum(q0 * q1, -1, keepdims=True) < 0, -q1, q1)
    q = q0 * (1 - w) + q1 * w
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def loop_crossfade(clip, P, max_blend_s=0.25):
    """Pick the end frame most similar to frame 0, then crossfade the tail into the head."""
    fps, Q, root = clip["fps"], clip["quats"], clip["root"]
    T = len(Q)
    lo = int(T * 0.6)
    D = np.array([np.linalg.norm((P[t] - P[t, :1]) - (P[0] - P[0, :1])) for t in range(lo, T)])
    e = lo + int(np.argmin(D)) if len(D) else T
    Q, root = Q[:e + 1].copy(), root[:e + 1].copy()
    n = min(int(max_blend_s * fps), len(Q) // 4)
    for k in range(n):
        w = ((k + 1) / n) ** 2 * (3 - 2 * (k + 1) / n) if n else 1      # smoothstep
        i = len(Q) - n + k
        Q[i] = nlerp(Q[i], Q[0], w)
        root[i, 1] = root[i, 1] * (1 - w) + root[0, 1] * w
    clip["quats"], clip["root"] = Q[:-1], root[:-1]      # last == first: drop duplicate
    if "root_track" in clip:
        clip["root_track"] = clip["root_track"][: len(Q) - 1]
    return clip


def resample(clip, tau):
    """Sample clip at fractional frame positions tau."""
    Q, root = clip["quats"], clip["root"]
    i0 = np.clip(np.floor(tau).astype(int), 0, len(Q) - 1)
    i1 = np.clip(i0 + 1, 0, len(Q) - 1)
    w = (tau - i0)[:, None]
    clip["root"] = root[i0] * (1 - w) + root[i1] * w
    if "root_track" in clip:
        rt = clip["root_track"]
        clip["root_track"] = rt[i0] * (1 - w) + rt[i1] * w
    clip["quats"] = nlerp(Q[i0], Q[i1], w[:, :, None] if Q.ndim == 3 else w)
    return clip


def stylize(clip, P, cfg):
    """KAIROS: play fast parts faster and slow parts slower (anticipation -> snap), x speed."""
    st = cfg["stylize"]
    fps = clip["fps"]
    e = motion_energy(P, fps)
    en = (e - np.percentile(e, 10)) / (np.percentile(e, 95) - np.percentile(e, 10) + 1e-6)
    rate = st["speed"] * (1 + st["snap"] * (np.clip(en, 0, 1.2) - 0.5) * 2)
    rate = np.clip(uniform_filter1d(rate, 5), 0.35 * st["speed"], 3 * st["speed"])
    tau = np.concatenate([[0], np.cumsum(rate[:-1])])
    tau = tau[tau <= len(P) - 1]
    clip = resample(clip, tau)
    clip["time_map"] = tau
    return clip


def hit_frames(P, fps, move, info):
    """Impact frames for attacks: peaks of hand/foot speed (count from manifest or name)."""
    tags = set(info.get("tags", []))
    if not (tags & {"attack", "hit_event"} or info.get("hits")):
        return []
    n = int(info.get("hits") or (move.split("combo_")[1].split("_")[0] if "combo_" in move else 1))
    idx = [I[j] for j in ("LeftHand", "RightHand", "LeftFoot", "RightFoot")]
    v = np.linalg.norm(np.gradient(P[:, idx], axis=0), axis=-1).max(1) * fps
    v = uniform_filter1d(v, 3)
    pk, prop = find_peaks(v, distance=max(1, int(0.12 * fps)), prominence=np.ptp(v) * 0.15)
    top = pk[np.argsort(-prop["prominences"])[:n]] if len(pk) else [int(np.argmax(v))]
    # the hit lands just after peak speed (extension), ~2 frames at 60fps
    return sorted(int(min(len(P) - 1, p + round(0.03 * fps))) for p in top)


# ---- main ---------------------------------------------------------------------------------------
def run(Pj, W, fps, cfg, move, info, aux=None):
    """Positions -> solved, cleaned clip dict ready for export, plus QC numbers."""
    Ps = smooth_positions(Pj, fps, cfg)
    sol = SV.solve(Ps, W, fps, aux)
    L, off = sol["lengths"], sol["offsets"]
    P = sol["fk"]
    clip = dict(fps=fps, root=sol["root"], quats=sol["quats"], lengths=L, offsets=off,
                tpose_frames=sol["tpose_frames"], src_index=np.arange(len(P)))
    qc = {"fit_residual_cm": float(np.nanmean(sol["residual"]) * 100)}
    # trim first so contacts/loops only see the performance
    if cfg["trim"]["enabled"] and not info.get("no_trim"):
        sl = trim(clip, P, cfg)
        P, clip["root"], clip["quats"] = P[sl], clip["root"][sl], clip["quats"][sl]
        clip["src_index"] = clip["src_index"][sl]
        qc["trim"] = [sl.start, sl.stop]
    con = foot_contacts(P, fps, cfg)
    qc["foot_slide_raw_cm"] = foot_slide_cm(P, con, fps)[0]
    if info.get("foot_lock", cfg["contacts"]["foot_lock"]):
        P = foot_lock(P, con, L, fps)
        re = SV.solve(P, np.ones(P.shape[:2]), fps, aux=None, lengths=L, offsets_=off)
        clip["quats"], clip["root"] = re["quats"], re["root"]
        P = re["fk"]
    qc["foot_slide_cm"], qc["foot_slide_max_cm"] = foot_slide_cm(P, foot_contacts(P, fps, cfg), fps)
    # root motion
    mode = info.get("root_motion", cfg["root_motion"])
    clip["root_motion_mode"] = mode
    clip["root_track"] = clip["root"].copy()      # world trajectory kept for previews / root bones
    if mode == "in_place":
        clip["root"] = clip["root"].copy()
        clip["root"][:, [0, 2]] = clip["root"][:, [0, 2]][:1] * 0
    else:
        clip["root"] = clip["root"] - np.array([clip["root"][0, 0], 0, clip["root"][0, 2]])
    if info.get("loop"):
        clip = loop_crossfade(clip, P)
        P = P[: len(clip["quats"])]
        clip["src_index"] = clip["src_index"][: len(P)]
    hits = hit_frames(P, fps, move, info)
    if cfg["stylize"]["enabled"]:
        clip = stylize(clip, P, cfg)
        tau = clip["time_map"]
        clip["src_index"] = clip["src_index"][np.clip(np.round(tau).astype(int), 0, len(P) - 1)]
        hits = [int(np.searchsorted(tau, h)) for h in hits]
    clip["hit_frames"] = hits
    clip["contacts"] = {s: c.tolist() for s, (c, _) in foot_contacts(P, fps, cfg).items()}
    return clip, qc
