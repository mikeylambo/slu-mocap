"""End-to-end synthetic tests:  python -m pytest -q tests   (or python tests/test_pipeline.py)"""
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.synth import walk, keypoints_from_joints, look_at, project, relative_to
from pipeline import skeleton as S, triangulate as TR, cleanup, export, qc, solve as SV
from pipeline.common import game_config, move_info

RNG = np.random.default_rng(1)


def _two_view(K3, noise_px=1.5, drop=0.03):
    tgt = [0, 0.9, 2.6]
    front = look_at([0.2, 1.1, 7.5], tgt)
    side = look_at([5.0, 1.0, 2.0], tgt)
    kps, cams = {}, {}
    for r, cam in (("leader", front), ("follower", side)):
        p = project(cam, K3) + RNG.normal(0, noise_px, K3.shape[:2] + (2,))
        c = np.full(K3.shape[:2], 0.9)
        c[RNG.random(c.shape) < drop] = 0.05
        kps[r] = (p, c)
        cams[r] = relative_to(front, cam)
    return cams, kps


def _pipeline(move="walk_loop", game="no_extraction", spin=False, **kw):
    root, Q, P, G = walk(spin=spin, **kw)
    K3 = keypoints_from_joints(P, G)
    cams, kps = _two_view(K3)
    X, conf, err = TR.triangulate(cams, kps)
    X, conf, _ = TR.fill_gaps(X, conf, 60)
    X, _ = TR.world_align(X, conf)
    Pj, W = S.joints_from_keypoints(X, conf)
    cfg = game_config(game)
    info = move_info(game, move)
    clip, q = cleanup.run(Pj, W, 60, cfg, move, info)
    return clip, q, err, cfg, info


def test_walk_two_view():
    clip, q, err, cfg, info = _pipeline()
    print("walk", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in q.items()})
    assert np.nanmedian(err) < 3.0
    assert q["foot_slide_cm"] < 2.0, q
    Pf, _ = SV.fk(clip["root_track"], clip["quats"], clip["lengths"], clip["offsets"])
    print("jitter mm", qc.jitter_mm(Pf)); assert qc.jitter_mm(Pf) < 3.0
    with tempfile.TemporaryDirectory() as d:
        meta = export.run(clip, Path(d), "no_extraction__walk_loop__t001", cfg, info, q)
        fr = export.read_bvh_frames(Path(d) / meta["files"]["bvh"])
        assert fr.shape == (len(clip["quats"]), 3 + 3 * (len(S.JOINTS) - len(S.END_SITES)))
        qc.contact_sheet(None, None, Pf[:60], Path(d) / "c.mp4", contacts=clip["contacts"])
        assert (Path(d) / "c.mp4").stat().st_size > 1000


def test_spin_stable():
    clip, q, *_ = _pipeline("turn_180", spin=True)
    Q = clip["quats"]
    # no 180-degree flips frame to frame
    d = np.abs(np.sum(Q[1:] * Q[:-1], -1))
    assert d.min() > 0.9, d.min()


def test_kairos_stylize_hits():
    clip, q, *_ = _pipeline("jab", game="kairos")
    assert clip["root_motion_mode"] == "in_place"
    assert np.allclose(clip["root"][:, [0, 2]], 0)
    assert len(clip["hit_frames"]) == 1



def bvh_fk(path):
    """Independent BVH parser + FK (validates channel order / offsets / units)."""
    from scipy.spatial.transform import Rotation as R
    txt = Path(path).read_text().split()
    joints, stack, i, cur = [], [], 0, None
    while txt[i] != "MOTION":
        tok = txt[i]
        if tok in ("ROOT", "JOINT"):
            joints.append({"name": txt[i + 1], "parent": stack[-1] if stack else None, "ch": 0}); cur = len(joints) - 1; i += 2
        elif tok == "End":
            joints.append({"name": "end", "parent": stack[-1], "ch": 0}); cur = len(joints) - 1; i += 2
        elif tok == "{":
            stack.append(cur); i += 1
        elif tok == "}":
            stack.pop(); i += 1
        elif tok == "OFFSET":
            joints[cur]["off"] = np.array(list(map(float, txt[i + 1:i + 4]))); i += 4
        elif tok == "CHANNELS":
            joints[cur]["ch"] = int(txt[i + 1]); i += 2 + joints[cur]["ch"]
        else:
            i += 1
    fr = export.read_bvh_frames(path)
    T = len(fr)
    pos = {}
    glob = {}
    c = 0
    for k, j in enumerate(joints):
        if j["ch"] == 6:
            t_, e = fr[:, c:c + 3], fr[:, c + 3:c + 6]; c += 6
        elif j["ch"] == 3:
            t_, e = None, fr[:, c:c + 3]; c += 3
        else:
            t_, e = None, None
        loc = R.from_euler("ZXY", e, degrees=True).as_matrix() if e is not None else np.eye(3)[None].repeat(T, 0)
        if j["parent"] is None:
            glob[k], pos[k] = loc, t_
        else:
            p = j["parent"]
            glob[k] = glob[p] @ loc
            pos[k] = pos[p] + np.einsum("tij,j->ti", glob[p], j["off"])
    return {j["name"]: pos[k] / 100 for k, j in enumerate(joints) if j["name"] != "end"}


def test_bvh_roundtrip():
    clip, q, err, cfg, info = _pipeline()
    with tempfile.TemporaryDirectory() as d:
        export.write_bvh(Path(d) / "a.bvh", clip)
        P = bvh_fk(Path(d) / "a.bvh")
    Pf, _ = SV.fk(clip["root"], clip["quats"], clip["lengths"], clip["offsets"])
    e = max(np.abs(P[j] - Pf[:, S.IDX[j]]).max() for j in P)
    assert e < 1e-3, e


if __name__ == "__main__":
    test_walk_two_view(); test_spin_stable(); test_kairos_stylize_hits(); test_bvh_roundtrip(); print("ALL OK")
