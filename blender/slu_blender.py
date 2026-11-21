"""Shared Blender helpers (run inside Blender or with the `bpy` wheel)."""
from __future__ import annotations
import re, sys
from pathlib import Path
import bpy
from mathutils import Matrix, Quaternion, Vector

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def argv():
    a = sys.argv
    return a[a.index("--") + 1:] if "--" in a else a[1:]


def load_yaml(p):
    import yaml
    return yaml.safe_load(open(p)) or {}


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 60


def import_bvh(path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=str(path), global_scale=0.01, frame_start=1, use_fps_scale=False,
                            update_scene_fps=False, update_scene_duration=False, rotate_mode="NATIVE",
                            axis_forward="-Z", axis_up="Y")
    ob = next(o for o in bpy.data.objects if o not in before)
    return ob


def import_model(path, keep_anim=False) -> bpy.types.Object:
    path = Path(path).expanduser()
    before = set(bpy.data.objects)
    if path.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif path.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False, ignore_leaf_bones=True)
    else:
        raise ValueError(f"unsupported model {path}")
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == "ARMATURE")
    for o in ([] if keep_anim else new):      # drop imported animations: we bake our own
        if o.animation_data:
            o.animation_data.action = None
    return arm


# ---- bone mapping ------------------------------------------------------------------------
CANON = [l.split()[0] for l in """Hips Spine Spine1 Spine2 Neck Head LeftShoulder LeftArm LeftForeArm LeftHand
RightShoulder RightArm RightForeArm RightHand LeftUpLeg LeftLeg LeftFoot LeftToeBase RightUpLeg RightLeg
RightFoot RightToeBase""".split()]
for _s in ("Left", "Right"):
    for _f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        CANON += [f"{_s}Hand{_f}{k}" for k in (1, 2, 3)]

ALIASES = {  # normalised synonyms -> canonical (side handled separately)
    "hips": "Hips", "pelvis": "Hips", "hip": "Hips", "spine": "Spine", "spine01": "Spine1", "spine1": "Spine1",
    "chest": "Spine1", "spine02": "Spine2", "spine2": "Spine2", "upperchest": "Spine2", "neck": "Neck",
    "neck01": "Neck", "head": "Head", "shoulder": "Shoulder", "clavicle": "Shoulder", "arm": "Arm",
    "upperarm": "Arm", "forearm": "ForeArm", "lowerarm": "ForeArm", "hand": "Hand", "upleg": "UpLeg",
    "thigh": "UpLeg", "upperleg": "UpLeg", "leg": "Leg", "calf": "Leg", "lowerleg": "Leg", "shin": "Leg",
    "foot": "Foot", "toebase": "ToeBase", "toe": "ToeBase", "toes": "ToeBase", "ball": "ToeBase",
}


def _norm(name):
    n = name.split(":")[-1].lower()
    side = None
    for pat, s in ((r"(^|[_.\s])l([_.\s]|$)|left", "Left"), (r"(^|[_.\s])r([_.\s]|$)|right", "Right")):
        if re.search(pat, n):
            side = s
            n = re.sub(pat, "_", n)
    n = re.sub(r"[^a-z0-9]", "", n)
    return side, n


def automap(names):
    """Guess canonical -> target names for a humanoid armature."""
    out = {}
    for t in names:
        side, n = _norm(t)
        m = re.match(r"(?:hand)?(thumb|index|middle|ring|pinky|little)(?:finger)?0?(\d)$", n)
        if m and side:
            f = "Pinky" if m[1] == "little" else m[1].capitalize()
            out.setdefault(f"{side}Hand{f}{int(m[2])}", t)
            continue
        c = ALIASES.get(n)
        if not c:
            continue
        key = (side + c) if c in ("Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "ToeBase") else c
        if side is None and key != c:
            continue
        if key in CANON:
            out.setdefault(key, t)
    return out


def rig_map(rig: dict, target_names) -> dict:
    m = dict(rig.get("map") or {})
    mir = rig.get("mirror")
    if mir:
        for k, v in list(m.items()):
            if k.startswith("Left"):
                rv = v.replace("Left", "Right") if mir is True else v[: -len(mir["from"])] + mir["to"] \
                    if v.endswith(mir["from"]) else v
                m.setdefault("Right" + k[4:], rv)
    pre = rig.get("prefix")
    if pre is not None:
        for c in CANON:
            m.setdefault(c, pre + c)
    if target_names is not None:
        tn = set(target_names)
        auto = automap(target_names)
        for c in CANON:
            if m.get(c) not in tn and c in auto:
                m[c] = auto[c]
        m = {k: v for k, v in m.items() if v in tn}
    return m


def world_rot(ob, pb_or_bone, rest=False):
    if rest:
        mat = ob.matrix_world @ pb_or_bone.matrix_local
    else:
        mat = ob.matrix_world @ pb_or_bone.matrix
    return mat.to_quaternion().normalized(), mat.to_translation()
