"""SLU canonical humanoid skeleton (Mixamo-compatible names) + COCO-WholeBody mapping.

Coordinate convention (world + rest pose): metres, Y up, character faces +Z, character's
left is +X. Rest pose is a T-pose, palms down.
"""
from __future__ import annotations
import numpy as np

# ---- COCO-WholeBody (133) indices used by RTMW ---------------------------------------
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSHO, RSHO, LELB, RELB, LWRI, RWRI = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16
LBTOE, LSTOE, LHEEL, RBTOE, RSTOE, RHEEL = 17, 18, 19, 20, 21, 22
LHAND0, RHAND0 = 91, 112          # 21 kps each: wrist, thumb1-4, index1-4, middle, ring, pinky
N_KP = 133

BODY_KP = list(range(0, 23))
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Body bones used for bone-length gap filling (kp index pairs)
KP_BONES = [(LSHO, LELB), (LELB, LWRI), (RSHO, RELB), (RELB, RWRI), (LHIP, LKNE), (LKNE, LANK),
            (RHIP, RKNE), (RKNE, RANK), (LANK, LHEEL), (LANK, LBTOE), (RANK, RHEEL), (RANK, RBTOE),
            (LSHO, RSHO), (LHIP, RHIP), (LSHO, LHIP), (RSHO, RHIP)]
for _h0 in (LHAND0, RHAND0):
    for _f in range(5):
        _chain = [_h0] + [_h0 + 1 + 4 * _f + k for k in range(4)]
        KP_BONES += list(zip(_chain[:-1], _chain[1:]))

# ---- Canonical joints ------------------------------------------------------------------
# (name, parent, rest direction from parent to this joint (unit, used for offsets))
_J = [
    ("Hips", None), ("Spine", "Hips"), ("Spine1", "Spine"), ("Spine2", "Spine1"),
    ("Neck", "Spine2"), ("Head", "Neck"), ("HeadTop_End", "Head"),
]
for s, sx in (("Left", 1), ("Right", -1)):
    _J += [(f"{s}Shoulder", "Spine2"), (f"{s}Arm", f"{s}Shoulder"), (f"{s}ForeArm", f"{s}Arm"),
           (f"{s}Hand", f"{s}ForeArm")]
    for f in FINGERS:
        _J += [(f"{s}Hand{f}1", f"{s}Hand"), (f"{s}Hand{f}2", f"{s}Hand{f}1"),
               (f"{s}Hand{f}3", f"{s}Hand{f}2"), (f"{s}Hand{f}4", f"{s}Hand{f}3")]
for s in ("Left", "Right"):
    _J += [(f"{s}UpLeg", "Hips"), (f"{s}Leg", f"{s}UpLeg"), (f"{s}Foot", f"{s}Leg"),
           (f"{s}ToeBase", f"{s}Foot"), (f"{s}Toe_End", f"{s}ToeBase")]

JOINTS = [j for j, _ in _J]
PARENT = {j: p for j, p in _J}
IDX = {j: i for i, j in enumerate(JOINTS)}
PARENT_IDX = np.array([IDX[p] if p else -1 for _, p in _J])
CHILDREN = {j: [c for c, p in _J if p == j] for j in JOINTS}
END_SITES = {j for j in JOINTS if not CHILDREN[j]}


def _n(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rest_dirs() -> dict[str, np.ndarray]:
    """Unit direction from parent joint to joint in the T-pose rest."""
    d = {"Hips": np.zeros(3), "Spine": _n([0, 1, 0]), "Spine1": _n([0, 1, 0]),
         "Spine2": _n([0, 1, 0]), "Neck": _n([0, 1, 0]), "Head": _n([0, 1, 0.08]),
         "HeadTop_End": _n([0, 1, 0])}
    for s, sx in (("Left", 1), ("Right", -1)):
        d[f"{s}Shoulder"] = _n([sx, 0.25, 0])
        d[f"{s}Arm"] = d[f"{s}ForeArm"] = d[f"{s}Hand"] = _n([sx, 0, 0])
        for f in FINGERS:
            base = {"Thumb": [sx, -0.3, 0.9], "Index": [sx, 0, 0.25], "Middle": [sx, 0, 0.05],
                    "Ring": [sx, 0, -0.12], "Pinky": [sx, 0, -0.3]}[f]
            d[f"{s}Hand{f}1"] = _n(base)
            tip = _n([sx, -0.2, 0.7]) if f == "Thumb" else _n([sx, 0, 0])
            for k in (2, 3, 4):
                d[f"{s}Hand{f}{k}"] = tip
        d[f"{s}UpLeg"] = _n([sx, -0.1, 0])
        d[f"{s}Leg"] = d[f"{s}Foot"] = _n([0, -1, 0])
        d[f"{s}ToeBase"] = _n([0, -0.45, 1])
        d[f"{s}Toe_End"] = _n([0, 0, 1])
    return d


REST_DIR = rest_dirs()

DEFAULT_LENGTHS = {  # metres, ~1.75 m adult; overwritten by T-pose estimate
    "Spine": 0.10, "Spine1": 0.12, "Spine2": 0.12, "Neck": 0.16, "Head": 0.10, "HeadTop_End": 0.18,
    "Shoulder": 0.17, "Arm": 0.28, "ForeArm": 0.26, "Hand": 0.09,
    "Thumb1": 0.04, "Thumb2": 0.035, "Thumb3": 0.03, "Thumb4": 0.025,
    "Finger1": 0.0, "Finger2": 0.045, "Finger3": 0.028, "Finger4": 0.022,
    "UpLeg": 0.10, "Leg": 0.44, "Foot": 0.43, "ToeBase": 0.15, "Toe_End": 0.05,
}


def default_length(j: str) -> float:
    k = j.replace("Left", "").replace("Right", "")
    if k.startswith("Hand") and len(k) > 4:
        f, n = k[4:-1], k[-1]
        return DEFAULT_LENGTHS[f"Thumb{n}" if f == "Thumb" else f"Finger{n}"] or 0.09
    return DEFAULT_LENGTHS.get(k, 0.1)


# ---- Joint positions from 133 keypoints -------------------------------------------------
def _hand_kp(side: str, finger: str, k: int) -> int:
    h0 = LHAND0 if side == "Left" else RHAND0
    return h0 + 1 + 4 * FINGERS.index(finger) + (k - 1)


def joints_from_keypoints(P: np.ndarray, C: np.ndarray):
    """P (T,133,3) world points, C (T,133) confidences -> (T,J,3) joint positions, (T,J) conf.

    Hand chains use the body wrist as the hand root so arm and hand always connect.
    """
    T = P.shape[0]
    J = np.full((T, len(JOINTS), 3), np.nan)
    W = np.zeros((T, len(JOINTS)))

    def put(name, pts, conf):
        J[:, IDX[name]] = pts
        W[:, IDX[name]] = conf

    mid = lambda a, b: 0.5 * (P[:, a] + P[:, b])
    cm = lambda *ix: np.min(np.stack([C[:, i] for i in ix]), 0)
    hips = mid(LHIP, RHIP)
    neck = mid(LSHO, RSHO)
    ears = mid(LEAR, REAR)
    up = neck - hips
    put("Hips", hips, cm(LHIP, RHIP))
    for k, name in enumerate(("Spine", "Spine1", "Spine2"), 1):
        put(name, hips + up * (k * 0.27), cm(LHIP, RHIP, LSHO, RSHO))
    put("Neck", neck + 0.12 * (ears - neck), cm(LSHO, RSHO))
    head = ears
    put("Head", head, cm(LEAR, REAR))
    upn = up / (np.linalg.norm(up, axis=-1, keepdims=True) + 1e-9)
    put("HeadTop_End", head + upn * 0.18, cm(LEAR, REAR))
    for s, sho, elb, wri, hip, kne, ank, bt, st, he in (
            ("Left", LSHO, LELB, LWRI, LHIP, LKNE, LANK, LBTOE, LSTOE, LHEEL),
            ("Right", RSHO, RELB, RWRI, RHIP, RKNE, RANK, RBTOE, RSTOE, RHEEL)):
        put(f"{s}Shoulder", neck + 0.35 * (P[:, sho] - neck), C[:, sho])
        put(f"{s}Arm", P[:, sho], C[:, sho])
        put(f"{s}ForeArm", P[:, elb], C[:, elb])
        put(f"{s}Hand", P[:, wri], C[:, wri])
        h0 = LHAND0 if s == "Left" else RHAND0
        for f in FINGERS:
            for k in (1, 2, 3, 4):
                i = _hand_kp(s, f, k)
                # translate hand so its wrist coincides with the body wrist
                put(f"{s}Hand{f}{k}", P[:, i] - P[:, h0] + P[:, wri], np.minimum(C[:, i], C[:, h0]))
        put(f"{s}UpLeg", P[:, hip], C[:, hip])
        put(f"{s}Leg", P[:, kne], C[:, kne])
        put(f"{s}Foot", P[:, ank], C[:, ank])
        ball = 0.5 * (P[:, bt] + P[:, st])
        put(f"{s}ToeBase", P[:, ank] + 0.72 * (ball - P[:, ank]), cm(bt, st))
        put(f"{s}Toe_End", ball, cm(bt, st))
    return J, W
