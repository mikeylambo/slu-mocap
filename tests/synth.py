"""Synthetic performer + virtual stereo rig for end-to-end pipeline tests (no video needed)."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from pipeline import skeleton as S, solve as SV

J, I = S.JOINTS, S.IDX
LEN = {j: S.default_length(j) for j in J if S.PARENT[j]}
LEN.update({f"{s}HandIndex1": 0.09 for s in ("Left", "Right")})
for s in ("Left", "Right"):
    for f in ("Middle", "Ring", "Pinky"):
        LEN[f"{s}Hand{f}1"] = 0.085


def walk(fps=60, seconds=4.0, speed=1.3, spin=False):
    """Procedural walk (or spin) cycle with a planted-foot root. Returns root, quats, positions."""
    T = int(fps * seconds)
    t = np.arange(T) / fps
    ph = 2 * np.pi * t * 0.9
    Q = np.tile([0, 0, 0, 1.0], (T, len(J), 1))
    def set_(j, rv):
        Q[:, I[j]] = R.from_rotvec(rv).as_quat()
    z = np.zeros(T)
    for s, sg in (("Left", 1), ("Right", -1)):
        p = ph if s == "Left" else ph + np.pi
        thigh = 0.45 * np.sin(p)
        knee = 0.6 * np.clip(np.sin(p - 1.2), 0, None) + 0.1
        set_(f"{s}UpLeg", np.stack([-thigh, z, z], 1))
        set_(f"{s}Leg", np.stack([knee, z, z], 1))                      # +X rot -> shin back
        set_(f"{s}Foot", np.stack([0.15 * np.sin(p), z, z], 1))
        set_(f"{s}Arm", np.stack([0.4 * np.sin(p + np.pi), z, -sg * np.full(T, 1.2)], 1))
        set_(f"{s}ForeArm", np.stack([z, -sg * (0.5 + 0.2 * np.sin(p)), z], 1))
        for f in S.FINGERS:
            for k in (1, 2, 3):
                set_(f"{s}Hand{f}{k}", np.stack([z, z, -sg * 0.3 * np.ones(T)], 1))
    set_("Spine1", np.stack([0.05 * np.ones(T), 0.1 * np.sin(ph), z], 1))
    yaw = 2 * np.pi * t / seconds * 2 if spin else z
    set_("Hips", np.stack([z, yaw, z], 1)) if spin else None
    root = np.stack([z, 0.95 + 0.02 * np.cos(2 * ph), speed * t * (0 if spin else 1)], 1)
    P, _ = SV.fk(root, Q, LEN)
    # glue the lowest foot to the floor
    feet = P[:, [I["LeftToe_End"], I["RightToe_End"], I["LeftFoot"], I["RightFoot"]], 1]
    root[:, 1] -= feet.min(1) - 0.0
    P, G = SV.fk(root, Q, LEN)
    return root, Q, P, G


def keypoints_from_joints(P, G):
    T = P.shape[0]
    K = np.full((T, S.N_KP, 3), np.nan)
    g = lambda j: P[:, I[j]]
    head, hR = g("Head"), G[:, I["Head"]]
    ax = lambda v: np.einsum("tij,j->ti", hR, np.array(v, float))
    K[:, S.NOSE] = head + ax([0, 0.0, 0.1]); K[:, S.LEYE] = head + ax([0.03, 0.03, 0.08])
    K[:, S.REYE] = head + ax([-0.03, 0.03, 0.08])
    K[:, S.LEAR] = head + ax([0.075, 0, 0]); K[:, S.REAR] = head + ax([-0.075, 0, 0])
    for s, sho, elb, wri, hip, kne, ank, bt, st, he, h0 in (
            ("Left", S.LSHO, S.LELB, S.LWRI, S.LHIP, S.LKNE, S.LANK, S.LBTOE, S.LSTOE, S.LHEEL, S.LHAND0),
            ("Right", S.RSHO, S.RELB, S.RWRI, S.RHIP, S.RKNE, S.RANK, S.RBTOE, S.RSTOE, S.RHEEL, S.RHAND0)):
        K[:, sho], K[:, elb], K[:, wri] = g(f"{s}Arm"), g(f"{s}ForeArm"), g(f"{s}Hand")
        K[:, hip], K[:, kne], K[:, ank] = g(f"{s}UpLeg"), g(f"{s}Leg"), g(f"{s}Foot")
        fR = G[:, I[f"{s}ToeBase"]]
        lat = np.einsum("tij,j->ti", fR, [0.035, 0, 0])
        K[:, bt] = g(f"{s}Toe_End") - lat * (1 if s == "Left" else -1)
        K[:, st] = g(f"{s}Toe_End") + lat * (1 if s == "Left" else -1)
        K[:, he] = g(f"{s}Foot") + np.einsum("tij,j->ti", G[:, I[f"{s}Foot"]], [0, -0.06, -0.06])
        K[:, h0] = g(f"{s}Hand")
        for fi, f in enumerate(S.FINGERS):
            for k in (1, 2, 3, 4):
                K[:, h0 + 1 + 4 * fi + k - 1] = g(f"{s}Hand{f}{k}")
    # keypoint mid-shoulder is what the solver calls "neck" base; fine for tests
    for k in range(S.N_KP):
        if np.isnan(K[:, k]).all():
            K[:, k] = K[:, S.NOSE]
    return K


def look_at(eye, target, f=1500.0, size=(1920, 1080)):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 1, 0]); x /= np.linalg.norm(x)   # world Y up -> cv y down
    y = np.cross(z, x)
    Rm = np.stack([x, y, z])
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    return {"K": K, "dist": np.zeros(5), "R": Rm, "t": -Rm @ eye, "size": list(size)}


def project(cam, X):
    c = X @ cam["R"].T + cam["t"]
    x = c @ cam["K"].T
    return x[..., :2] / x[..., 2:3]


def relative_to(ref, cam):
    """Express cam extrinsics relative to ref camera (pipeline convention: ref at origin)."""
    R_ = cam["R"] @ ref["R"].T
    t_ = cam["t"] - R_ @ ref["t"]
    return {**cam, "R": R_, "t": t_}
