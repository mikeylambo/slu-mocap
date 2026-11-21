"""Retarget canonical SLU BVH clips onto game rigs (headless Blender).

    blender -b -P blender/retarget.py -- --rig mixamo --out OUT [--model rig.glb] [--batch NAME] a.bvh b.bvh ...

* Rest-pose independent: per-bone world rotation deltas from the source rest, applied to the
  target bone after it is auto-aligned to the source T-pose (works for A-pose rigs).
* Hips translation scaled by leg length; optional root bone receives ground-plane motion.
* Writes <clip>.glb + <clip>.fbx per clip, and <batch>.glb with every clip as a named action.
* No --model: exports the canonical skeleton with the rig's bone names (Mixamo / UE / Godot).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import bpy
from mathutils import Matrix, Quaternion, Vector
sys_path = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(sys_path))
from slu_blender import (REPO, argv, load_yaml, reset, import_bvh, import_model, rig_map, world_rot, automap)


def leg_length(ob, names):
    b = ob.data.bones
    try:
        up, lo, ft = (b[names[k]] for k in ("LeftUpLeg", "LeftLeg", "LeftFoot"))
    except KeyError:
        return None
    mw = ob.matrix_world
    return ((mw @ lo.head_local) - (mw @ up.head_local)).length + ((mw @ ft.head_local) - (mw @ lo.head_local)).length


def tpose_align(src, tgt, cmap):
    """World rest rotation of each mapped target bone, swung so its direction matches the source."""
    out = {}
    sb, tb = src.data.bones, tgt.data.bones
    inv = {v: k for k, v in cmap.items()}
    for c, t in cmap.items():
        bt = tb[t]
        q, _ = world_rot(tgt, bt, rest=True)
        # direction to first mapped child (both rigs)
        kids = [k for k in bt.children_recursive if k.name in inv]
        kid = min(kids, key=lambda k: len(k.parent_recursive)) if kids else None
        if kid is not None and inv[kid.name] in sb and c in sb:
            mw_t, mw_s = tgt.matrix_world, src.matrix_world
            dt = (mw_t @ kid.head_local) - (mw_t @ bt.head_local)
            ds = (mw_s @ sb[inv[kid.name]].head_local) - (mw_s @ sb[c].head_local)
            if dt.length > 1e-6 and ds.length > 1e-6:
                q = dt.normalized().rotation_difference(ds.normalized()) @ q
        out[t] = q
    return out


def bake(src, tgt, cmap, action_name, root_bone=None, frames=None):
    scn = bpy.context.scene
    f0, f1 = frames or (int(src.animation_data.action.frame_range[0]), int(src.animation_data.action.frame_range[1]))
    tpose = tpose_align(src, tgt, cmap)
    src_rest = {c: world_rot(src, src.data.bones[c], rest=True)[0] for c in cmap if c in src.data.bones}
    hips_c = "Hips"
    s_hips_rest = world_rot(src, src.data.bones[hips_c], rest=True)[1]
    ls, lt = leg_length(src, {k: k for k in cmap}), leg_length(tgt, cmap)
    scale = (lt / ls) if ls and lt else 1.0
    tgt.animation_data_create()
    act = bpy.data.actions.new(action_name)
    tgt.animation_data.action = act
    order = []
    def walk(b):
        order.append(b)
        for c in b.children:
            walk(c)
    for b in tgt.data.bones:
        if b.parent is None:
            walk(b)
    inv_mw = tgt.matrix_world.inverted()
    tw_rot = tgt.matrix_world.to_quaternion()
    for pb in tgt.pose.bones:
        pb.rotation_mode = "QUATERNION"
    for f in range(f0, f1 + 1):
        scn.frame_set(f)
        desired = {}
        for b in order:
            pb = tgt.pose.bones[b.name]
            par = b.parent
            rest_local = (par.matrix_local.inverted() @ b.matrix_local).to_quaternion() if par else \
                b.matrix_local.to_quaternion()
            par_des = desired[par.name] if par else tw_rot
            c = next((k for k, v in cmap.items() if v == b.name), None)
            if c is not None and c in src_rest:
                qs, _ = world_rot(src, src.pose.bones[c])
                delta = qs @ src_rest[c].inverted()
                des = delta @ tpose[b.name]
            else:
                des = par_des @ rest_local
            desired[b.name] = des
            basis = rest_local.inverted() @ par_des.inverted() @ des
            pb.rotation_quaternion = basis
            pb.keyframe_insert("rotation_quaternion", frame=f, group=b.name)
        # translation: hips (and optional root bone for ground-plane motion)
        _, sp = world_rot(src, src.pose.bones[hips_c])
        d = (sp - s_hips_rest) * scale
        hips_t = tgt.pose.bones[cmap[hips_c]]
        move = d.copy()
        if root_bone and root_bone in tgt.pose.bones:
            ground = Vector((d.x, d.y, 0.0))         # Blender world is Z-up
            move = Vector((0.0, 0.0, d.z))
            rb = tgt.pose.bones[root_bone]
            rb.location = _world_to_basis_loc(tgt, rb, ground)
            rb.keyframe_insert("location", frame=f, group=root_bone)
        hips_t.location = _world_to_basis_loc(tgt, hips_t, move)
        hips_t.keyframe_insert("location", frame=f, group=hips_t.name)
    return act


def _world_to_basis_loc(ob, pb, world_delta):
    """World-space translation delta -> pose-bone location (bone rest space)."""
    m = (ob.matrix_world @ pb.bone.matrix_local).to_3x3().normalized()
    return m.inverted() @ world_delta


def rename_only(src, cmap):
    """No target model: rename the canonical armature's bones to the rig's names."""
    for c, t in cmap.items():
        if c in src.data.bones:
            src.data.bones[c].name = t
    return src


def export(ob, path_glb=None, path_fbx=None, all_actions=False, name="Animation"):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    for ch in ob.children:
        ch.select_set(True)
    bpy.context.view_layer.objects.active = ob
    if path_glb:
        kw = dict(filepath=str(path_glb), export_format="GLB", use_selection=True, export_animations=True,
                  export_animation_mode="ACTIONS" if all_actions else "ACTIVE_ACTIONS",
                  export_force_sampling=True, export_frame_range=False, export_anim_slide_to_zero=True,
                  export_optimize_animation_size=False)
        try:   # Blender 4.3+: name the single exported animation after the move
            bpy.ops.export_scene.gltf(**kw, export_merged_animation_name=name)
        except TypeError:
            bpy.ops.export_scene.gltf(**kw)
    if path_fbx:
        bpy.ops.export_scene.fbx(filepath=str(path_fbx), use_selection=True, add_leaf_bones=False,
                                 bake_anim=True, bake_anim_use_all_actions=False,
                                 bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0.0,
                                 object_types={"ARMATURE", "MESH"})


def to_nla(ob, act):
    tr = ob.animation_data.nla_tracks.new()
    tr.name = act.name
    st = tr.strips.new(act.name, int(act.frame_range[0]), act)
    st.name = act.name
    tr.mute = False
    ob.animation_data.action = None


def import_source(path: Path):
    """Canonical SLU BVH, or third-party BVH/FBX (e.g. Rokoko Vision) renamed via automap."""
    src = import_bvh(path) if path.suffix.lower() == ".bvh" else import_model(path, keep_anim=True)
    if "Hips" not in src.data.bones:
        m = automap([b.name for b in src.data.bones])
        for c, t in m.items():
            src.data.bones[t].name = c
        # rename fcurve paths happen automatically for bones renamed via the API
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--rig", default="mixamo")
    ap.add_argument("--model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", help="also write <out>/<batch>.glb with every clip as an action")
    ap.add_argument("--no-fbx", action="store_true")
    a = ap.parse_args(argv())
    rig = load_yaml(REPO / f"config/rigs/{a.rig}.yaml")
    model = a.model or rig.get("model")
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    reset()
    tgt = import_model(model) if model else None
    names = [b.name for b in tgt.data.bones] if tgt else None
    for clip in map(Path, a.clips):
        name = clip.stem.split("__")[1] if "__" in clip.stem else clip.stem
        src = import_source(clip)
        if tgt is None:
            cmap = rig_map(rig, None)
            ob = rename_only(src, cmap)
            ob.animation_data.action.name = name
            ob.name = f"{a.rig}_{name}"
        else:
            cmap = rig_map(rig, names)
            missing = [c for c in ("Hips", "LeftUpLeg", "LeftArm", "Head") if c not in cmap]
            if missing:
                raise SystemExit(f"retarget: rig {a.rig} has no bones for {missing}; add them to config/rigs/{a.rig}.yaml map")
            ob = tgt
            bake(src, tgt, cmap, name, rig.get("root_bone"))
        glb, fbx = out / f"{clip.stem}.{a.rig}.glb", out / f"{clip.stem}.{a.rig}.fbx"
        export(ob, glb, None if a.no_fbx else fbx, name=name)
        results.append({"clip": clip.name, "action": name, "glb": glb.name, "fbx": None if a.no_fbx else fbx.name,
                        "bones_mapped": len(cmap)})
        if a.batch:
            to_nla(ob, ob.animation_data.action)
        if tgt is None and not a.batch:
            bpy.data.objects.remove(src, do_unlink=True)
        elif tgt is not None:
            sa = src.animation_data.action
            bpy.data.objects.remove(src, do_unlink=True)
            bpy.data.actions.remove(sa)
    if a.batch:
        if tgt is None:   # merge renamed canonical armatures: keep first, move other actions over
            arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
            keep = arms[0]
            for o in arms[1:]:
                for tr in o.animation_data.nla_tracks:
                    to_nla_action = tr.strips[0].action
                    t2 = keep.animation_data.nla_tracks.new(); t2.name = tr.name
                    t2.strips.new(tr.name, int(to_nla_action.frame_range[0]), to_nla_action)
                bpy.data.objects.remove(o, do_unlink=True)
            ob = keep
        export(ob, out / f"{a.batch}.glb", None, all_actions=True)
        results.append({"batch": f"{a.batch}.glb", "actions": [r["action"] for r in results]})
    (out / f"retarget_{a.rig}.json").write_text(json.dumps(results, indent=2))
    print("RETARGET_OK", json.dumps(results))


if __name__ == "__main__":
    main()
