"""export: BVH (cm, ZXY, 60 fps, Mixamo bone names) + FBX via headless Blender + clip JSON."""
from __future__ import annotations
import shutil, subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from . import skeleton as S
from .common import REPO, log, write_json

CM = 100.0


def write_bvh(path: Path, clip: dict):
    off, Q, root, fps = clip["offsets"], clip["quats"], clip["root"], clip["fps"]
    lines = ["HIERARCHY"]

    def emit(j, depth):
        ind = "\t" * depth
        o = off[j] * CM
        if S.PARENT[j] is None:
            lines.append(f"ROOT {j}")
        elif j in S.END_SITES:
            lines.extend([f"{ind}End Site", f"{ind}{{", f"{ind}\tOFFSET {o[0]:.4f} {o[1]:.4f} {o[2]:.4f}", f"{ind}}}"])
            return
        else:
            lines.append(f"{ind}JOINT {j}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}\tOFFSET {o[0]:.4f} {o[1]:.4f} {o[2]:.4f}")
        ch = "6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" if S.PARENT[j] is None \
            else "3 Zrotation Xrotation Yrotation"
        lines.append(f"{ind}\tCHANNELS {ch}")
        for c in S.CHILDREN[j]:
            emit(c, depth + 1)
        lines.append(f"{ind}}}")

    emit("Hips", 0)
    chan = [j for j in S.JOINTS if j not in S.END_SITES]
    T = len(Q)
    eul = np.stack([R.from_quat(Q[:, S.IDX[j]]).as_euler("ZXY", degrees=True) for j in chan], 1)  # T,C,3
    eul = np.unwrap(np.radians(eul), axis=0) * 180 / np.pi
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.7f}"]
    for t in range(T):
        row = list(root[t] * CM) + list(eul[t].ravel())
        lines.append(" ".join(f"{v:.4f}" for v in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_bvh_frames(path):
    """Tiny reader for tests: returns (joint names with channels, frames array)."""
    txt = Path(path).read_text().split("MOTION")[1].strip().splitlines()
    return np.array([[float(v) for v in l.split()] for l in txt[2:]])


def find_blender() -> str | None:
    for c in (shutil.which("blender"), "/Applications/Blender.app/Contents/MacOS/Blender",
              "/content/blender/blender", str(Path.home() / "blender/blender")):
        if c and Path(c).exists():
            return c
    return None


def bvh_to_fbx(bvh: Path, fbx: Path) -> bool:
    b = find_blender()
    if not b:
        return False
    r = subprocess.run([b, "-b", "--factory-startup", "-P", str(REPO / "blender/convert.py"), "--",
                        str(bvh), str(fbx)], capture_output=True, text=True)
    if r.returncode != 0 or not fbx.exists():
        log("export", f"FBX failed: {r.stderr[-400:]}")
        return False
    return True


def run(clip: dict, out_dir: Path, take_id: str, cfg: dict, info: dict, qc: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bvh = out_dir / f"{take_id}.bvh"
    write_bvh(bvh, clip)
    files = {"bvh": bvh.name}
    if cfg["export"].get("fbx") and bvh_to_fbx(bvh, out_dir / f"{take_id}.fbx"):
        files["fbx"] = f"{take_id}.fbx"
    meta = {"take": take_id, "fps": clip["fps"], "frames": len(clip["quats"]),
            "loop": bool(info.get("loop")), "root_motion": clip.get("root_motion_mode"),
            "hit_frames": clip.get("hit_frames", []), "tags": info.get("tags", []),
            "notes": info.get("notes", ""), "files": files, "qc": qc,
            "lengths_m": {k: round(v, 4) for k, v in clip["lengths"].items()}}
    write_json(out_dir / f"{take_id}.clip.json", meta)
    np.savez_compressed(out_dir / f"{take_id}.anim.npz", root=clip["root"], quats=clip["quats"],
                        root_track=clip.get("root_track", clip["root"]))
    return meta
