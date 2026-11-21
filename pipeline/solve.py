"""solve: fit the SLU canonical skeleton to 3D joint targets.

* constant bone lengths (from the T-pose window if present, else robust clip median; L/R symmetric)
* per-bone orientation from a primary direction + a secondary (twist) reference
* hinge joints (elbows, knees) solved in their bend plane, clamped to [0, 160] deg (joint limits)
* bend-plane degeneracy (straight limbs) blends smoothly to the parent's frame
* output: root translation + local quaternions, FK positions, per-joint fit residual
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from . import skeleton as S

J = S.JOINTS
I = S.IDX
HINGES = {"LeftForeArm", "RightForeArm", "LeftLeg", "RightLeg"}
HINGE_MAX = np.radians(160)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


# ---- bone lengths ------------------------------------------------------------------------
def tpose_window(Pj, W, fps):
    """Frames where both arms are extended ~horizontally and still (T-pose)."""
    def straight_horiz(s):
        a, f, h = Pj[:, I[f"{s}Arm"]], Pj[:, I[f"{s}ForeArm"]], Pj[:, I[f"{s}Hand"]]
        d = _unit(h - a)
        straight = np.einsum("ti,ti->t", _unit(f - a), _unit(h - f)) > 0.9
        return straight & (np.abs(d[:, 1]) < 0.35)
    m = straight_horiz("Left") & straight_horiz("Right")
    v = np.linalg.norm(np.gradient(Pj[:, I["LeftHand"]], axis=0), axis=-1) * fps
    m &= np.nan_to_num(v, nan=9) < 0.4
    return m


def bone_lengths(Pj, W, fps, conf_thr=0.5):
    tw = tpose_window(Pj, W, fps)
    use_t = tw.sum() >= int(0.3 * fps)
    L = {}
    for j in J:
        p = S.PARENT[j]
        if p is None:
            continue
        d = np.linalg.norm(Pj[:, I[j]] - Pj[:, I[p]], axis=-1)
        ok = np.isfinite(d) & (np.minimum(W[:, I[j]], W[:, I[p]]) > conf_thr)
        if use_t and (ok & tw).sum() > 5:
            ok &= tw
        L[j] = float(np.median(d[ok])) if ok.sum() > 5 else S.default_length(j)
    for j in J:  # symmetrise
        if j.startswith("Left"):
            r = "Right" + j[4:]
            L[j] = L[r] = 0.5 * (L[j] + L[r])
    return L, (np.where(tw)[0] if use_t else np.array([], int))


def rest_positions(L):
    P = np.zeros((len(J), 3))
    for j in J:
        p = S.PARENT[j]
        if p is not None:
            P[I[j]] = P[I[p]] + S.REST_DIR[j] * L[j]
    return P


def offsets(L):
    return {j: (S.REST_DIR[j] * L[j] if S.PARENT[j] else np.zeros(3)) for j in J}


def enforce_lengths(Pj, L):
    """Rebuild positions top-down keeping target directions but fixed bone lengths."""
    Q = Pj.copy()
    for j in J:
        p = S.PARENT[j]
        if p is None:
            continue
        d = _unit(Pj[:, I[j]] - Pj[:, I[p]])
        d = np.where(np.isfinite(d), d, S.REST_DIR[j])
        Q[:, I[j]] = Q[:, I[p]] + d * L[j]
    return Q


# ---- orientation frames --------------------------------------------------------------------
def _basis(a, b):
    """Orthonormal frame with x=a, y from b (Gram-Schmidt). Returns (...,3,3) columns."""
    x = _unit(a)
    y = _unit(b - x * np.sum(b * x, -1, keepdims=True))
    z = np.cross(x, y)
    return np.stack([x, y, z], -1)


def _frame_rot(a, b, a0, b0):
    """Rotation mapping rest vectors (a0,b0) to observed (a,b)."""
    B = _basis(a, b)
    B0 = _basis(np.broadcast_to(a0, a.shape), np.broadcast_to(b0, b.shape))
    return B @ np.swapaxes(B0, -1, -2)


def _bend_rest(side, limb):
    a0 = S.REST_DIR[f"{side}{'ForeArm' if limb == 'arm' else 'Leg'}"]
    bend = np.array([0, 0, 1.0]) if limb == "arm" else np.array([0, 0, -1.0])
    return _unit(np.cross(a0, bend))


def global_rotations(P, aux=None):
    """P (T,J,3) length-consistent positions -> {joint: (T,3,3)} global bone rotations."""
    T = P.shape[0]
    G = {}
    p = lambda n: P[:, I[n]]
    rp = rest_positions({j: 1.0 for j in J})  # direction-only rest (lengths irrelevant)
    r = lambda n: rp[I[n]]
    hip_x, sho_x = p("LeftUpLeg") - p("RightUpLeg"), p("LeftArm") - p("RightArm")
    up_t = p("Spine2") - p("Hips")
    G["Hips"] = _frame_rot(hip_x, up_t, r("LeftUpLeg") - r("RightUpLeg"), np.array([0, 1.0, 0]))
    rest_x = np.array([1.0, 0, 0])
    for name, child, k in (("Spine", "Spine1", 0.33), ("Spine1", "Spine2", 0.66), ("Spine2", "Neck", 1.0)):
        side = _unit(hip_x) * (1 - k) + _unit(sho_x) * k
        G[name] = _frame_rot(p(child) - p(name), side, S.REST_DIR[child], rest_x)
    head_x = aux.get("head_x") if aux else None
    head_f = aux.get("head_fwd") if aux else None
    if head_x is None:
        head_x, head_f = sho_x, np.cross(sho_x, up_t)
    G["Neck"] = _frame_rot(p("Head") - p("Neck"), _unit(sho_x) + _unit(head_x), S.REST_DIR["Head"], rest_x)
    G["Head"] = _frame_rot(head_x, head_f, rest_x, np.array([0, 0, 1.0]))
    for s in ("Left", "Right"):
        G[f"{s}Shoulder"] = _frame_rot(p(f"{s}Arm") - p(f"{s}Shoulder"), up_t,
                                       S.REST_DIR[f"{s}Arm"], np.array([0, 1.0, 0]))
        for limb, a, b, c, parent in (("arm", f"{s}Arm", f"{s}ForeArm", f"{s}Hand", f"{s}Shoulder"),
                                      ("leg", f"{s}UpLeg", f"{s}Leg", f"{s}Foot", "Hips")):
            u, l = p(b) - p(a), p(c) - p(b)
            n0 = _bend_rest(s, limb)
            n_raw = np.cross(_unit(u), _unit(l))
            sinang = np.linalg.norm(n_raw, axis=-1, keepdims=True)
            # fallback normal: parent frame carries the rest normal (limb straight)
            n_par = np.einsum("tij,j->ti", G[parent], n0)
            if limb == "leg":   # knees point where the feet point when straight
                fwd = p(f"{s}ToeBase") - p(c)
                n_ft = np.cross(_unit(u), _unit(fwd))
                n_par = np.where(np.linalg.norm(n_ft, axis=-1, keepdims=True) > 0.3, -_unit(n_ft), n_par)
            n_raw = _unit(n_raw)
            # joint limit: no hyper-extension -> flip normals pointing against the parent's
            flip = np.sum(n_raw * n_par, -1, keepdims=True) < -0.2
            w = np.clip((sinang - 0.08) / 0.2, 0, 1) * ~flip
            n = _unit(w * n_raw + (1 - w) * n_par)
            G[a] = _frame_rot(u, n, S.REST_DIR[b], n0)
            G[b] = _frame_rot(l, n, S.REST_DIR[c], n0)
        # hand: middle knuckle + index->pinky line
        palm = p(f"{s}HandIndex1") - p(f"{s}HandPinky1")
        G[f"{s}Hand"] = _frame_rot(p(f"{s}HandMiddle1") - p(f"{s}Hand"), palm,
                                   S.REST_DIR[f"{s}HandMiddle1"], r(f"{s}HandIndex1") - r(f"{s}HandPinky1"))
        palm0 = r(f"{s}HandIndex1") - r(f"{s}HandPinky1")
        for f in S.FINGERS:
            for k in (1, 2, 3):
                a, b = f"{s}Hand{f}{k}", f"{s}Hand{f}{k+1}"
                G[a] = _frame_rot(p(b) - p(a), palm, S.REST_DIR[b], palm0)
        # foot / toes: forward + up-from-shin
        up_leg = p(f"{s}Leg") - p(f"{s}Foot")
        G[f"{s}Foot"] = _frame_rot(p(f"{s}ToeBase") - p(f"{s}Foot"), up_leg,
                                   S.REST_DIR[f"{s}ToeBase"], np.array([0, 1.0, 0]))
        G[f"{s}ToeBase"] = _frame_rot(p(f"{s}Toe_End") - p(f"{s}ToeBase"), up_leg,
                                      S.REST_DIR[f"{s}Toe_End"], np.array([0, 1.0, 0]))
    for j in J:
        G.setdefault(j, None)
    return G


def local_quats(G):
    """{joint: (T,3,3)} global -> (T,J,4) local quats (xyzw), hinge-limited, sign-continuous."""
    T = next(v for v in G.values() if v is not None).shape[0]
    Q = np.tile([0, 0, 0, 1.0], (T, len(J), 1))
    eff = {}
    for j in J:
        g = G[j]
        par = S.PARENT[j]
        if g is None:                 # end sites: inherit parent's orientation
            eff[j] = eff[par]
            continue
        pg = eff[par] if par else np.eye(3)[None]
        loc = np.swapaxes(pg, -1, -2) @ g
        rot = R.from_matrix(loc)
        if j in HINGES:
            side = "Left" if j.startswith("Left") else "Right"
            axis = _bend_rest(side, "arm" if "Fore" in j else "leg")
            rv = rot.as_rotvec()
            ang = np.clip(rv @ axis, 0, HINGE_MAX)
            rot = R.from_rotvec(axis[None] * ang[:, None])
            loc = rot.as_matrix()
        eff[j] = pg @ loc
        q = rot.as_quat()
        # hemisphere continuity
        dots = np.sum(q[1:] * q[:-1], -1)
        flips = np.cumsum(np.r_[0, dots < 0]) % 2 == 1
        q[flips] *= -1
        Q[:, I[j]] = q
    return Q


def fit_offsets(P, G, L):
    """Actor-specific rest offsets: median child direction in the parent's bone frame.

    Branch joints (hips->legs, chest->shoulders/neck, hand->fingers) are not pinned by the
    parent's own frame, so their offsets are measured from the performance itself.
    """
    off = offsets(L)
    for j in J:
        par = S.PARENT[j]
        if par is None or G.get(par) is None:
            continue
        d = _unit(P[:, I[j]] - P[:, I[par]])
        loc = np.einsum("tji,tj->ti", G[par], d)          # G^T d
        loc = loc[np.isfinite(loc).all(-1)]
        if len(loc):
            m = np.median(loc, 0)
            off[j] = _unit(m) * L[j]
    return off


def fk(root, Q, L, off=None):
    """root (T,3), Q (T,J,4) local, L lengths -> positions (T,J,3), globals (T,J,3,3)."""
    T = root.shape[0]
    Pg = np.zeros((T, len(J), 3))
    Gm = np.zeros((T, len(J), 3, 3))
    off = off or offsets(L)
    for j in J:
        i = I[j]
        loc = R.from_quat(Q[:, i]).as_matrix()
        par = S.PARENT[j]
        if par is None:
            Gm[:, i] = loc
            Pg[:, i] = root
        else:
            pi = I[par]
            Gm[:, i] = Gm[:, pi] @ loc
            Pg[:, i] = Pg[:, pi] + np.einsum("tij,j->ti", Gm[:, pi], off[j])
    return Pg, Gm


def solve(Pj, W, fps, aux=None, lengths=None, offsets_=None):
    """Main entry. Returns dict(root, quats, lengths, fk, residual, tpose_frames)."""
    Pj = Pj.copy()
    for k in range(Pj.shape[1]):     # any fully-missing joint -> parent + rest
        if not np.isfinite(Pj[:, k]).any():
            par = S.PARENT[J[k]]
            Pj[:, k] = Pj[:, I[par]] + S.REST_DIR[J[k]] * S.default_length(J[k])
    if lengths is None:
        lengths, tpf = bone_lengths(Pj, W, fps)
    else:
        tpf = np.array([], int)
    Pc = enforce_lengths(Pj, lengths)
    G = global_rotations(Pc, aux)
    Q = local_quats(G)
    off = fit_offsets(Pc, G, lengths) if offsets_ is None else offsets_
    root = Pc[:, I["Hips"]]
    Pfk, _ = fk(root, Q, lengths, off)
    res = np.linalg.norm(Pfk - Pj, axis=-1)
    return dict(root=root, quats=Q, lengths=lengths, offsets=off, fk=Pfk, residual=res,
                tpose_frames=tpf)
