"""KAIROS sprite renderer: clip -> orthographic side view -> trimmed sprite sheet + JSON atlas.

    blender -b -P blender/render_sprites.py -- --game kairos --out OUT a.bvh [b.bvh ...]

* Zenith model from config/games/kairos.yaml (sprites.model); falls back to a capsule mannequin
  so the pipeline runs before the model exists.
* Toon (banded light, flat colours) + inverted-hull outline, built from emission shaders so it
  renders identically in Cycles on a GPU-less Colab/Mac (no display needed) and in EEVEE.
* Every Nth frame (sprites.frame_step), cells trimmed to the alpha bounding box, shelf-packed.
* Atlas: rects, trim offsets, pivots (feet / root on the floor), per-frame root motion, events.
"""
from __future__ import annotations
import argparse, json, math, sys, tempfile
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slu_blender import REPO, argv, load_yaml, reset, import_bvh, import_model, rig_map
import retarget as RT


def game_cfg(game):
    sys.path.insert(0, str(REPO))
    from pipeline.common import game_config
    return game_config(game)


# ---- look -------------------------------------------------------------------------------------
def toon_material(base, bands=3, light=(0.4, -0.5, 0.75), shadow=0.55):
    """Banded-diffuse toon as an emission shader (engine independent, light-rig free)."""
    m = bpy.data.materials.new("slu_toon")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = Vector(light).normalized()
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (shadow, shadow, shadow * 1.05, 1)
    els[1].position, els[1].color = 0.55, (1, 1, 1, 1)
    if bands > 2:
        e = els.new(0.2); e.color = (0.78, 0.78, 0.8, 1)
    mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
    mul.inputs[0].default_value = 1.0
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(geo.outputs["Normal"], dot.inputs[0])
    nt.links.new(dot.outputs["Value"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], mul.inputs[6])
    if isinstance(base, bpy.types.NodeSocket):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = base.links[0].from_node.image if base.is_linked and hasattr(base.links[0].from_node, "image") else None
        nt.links.new(tex.outputs[0], mul.inputs[7])
    else:
        mul.inputs[7].default_value = base
    nt.links.new(mul.outputs[2], em.inputs[0])
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


def toonify(mesh_obs):
    for ob in mesh_obs:
        for slot in ob.material_slots:
            src = slot.material
            base = (0.8, 0.8, 0.82, 1)
            if src and src.use_nodes:
                bsdf = next((n for n in src.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
                if bsdf:
                    bc = bsdf.inputs["Base Color"]
                    base = bc if bc.is_linked else tuple(bc.default_value)
            slot.material = toon_material(base)
        if not ob.material_slots:
            ob.data.materials.append(toon_material((0.85, 0.85, 0.88, 1)))


def outline(mesh_obs, width_m):
    m = bpy.data.materials.new("slu_outline")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    em = nt.nodes.new("ShaderNodeEmission"); em.inputs[0].default_value = (0.02, 0.02, 0.03, 1)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(geo.outputs["Backfacing"], mix.inputs[0])
    nt.links.new(em.outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    m.use_backface_culling = True
    for ob in mesh_obs:
        ob.data.materials.append(m)
        sol = ob.modifiers.new("slu_outline", "SOLIDIFY")
        sol.thickness = -width_m
        sol.offset = 1.0
        sol.use_flip_normals = True
        sol.material_offset = len(ob.data.materials) - 1
        sol.use_rim = False


def mannequin(arm):
    """Capsule body on the canonical armature (placeholder until the real model exists)."""
    obs = []
    for b in arm.data.bones:
        L = b.length
        if L < 0.03 or any(k in b.name for k in ("Thumb", "Index", "Middle", "Ring", "Pinky", "_End")):
            continue
        r = 0.11 if b.name in ("Hips", "Spine", "Spine1", "Spine2") else 0.09 if b.name == "Head" else \
            0.06 if "UpLeg" in b.name or ("Leg" in b.name and "Up" not in b.name) else 0.045
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=L, vertices=16)
        ob = bpy.context.object
        ob.data.transform(__import__("mathutils").Matrix.Translation((0, 0, L / 2)))
        ob.rotation_euler = (math.radians(-90), 0, 0)   # cylinder z -> bone y
        ob.data.transform(ob.matrix_basis)
        ob.rotation_euler = (0, 0, 0)
        bpy.ops.object.shade_smooth()
        ob.parent = arm
        ob.parent_type = "BONE"
        ob.parent_bone = b.name
        ob.matrix_parent_inverse.identity()
        ob.location = (0, -L, 0)                         # bone-parent origin is the tail
        obs.append(ob)
    return obs


# ---- render + pack ---------------------------------------------------------------------------
def setup_scene(cell, ppm, facing, engine):
    scn = bpy.context.scene
    scn.render.engine = engine
    if engine == "CYCLES":
        scn.cycles.samples = 8
        scn.cycles.use_denoising = False
        scn.cycles.device = "CPU"
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            if any(d.type == "CUDA" for d in prefs.devices):
                for d in prefs.devices:
                    d.use = True
                scn.cycles.device = "GPU"
        except Exception:
            pass
    scn.render.film_transparent = True
    scn.render.resolution_x = scn.render.resolution_y = cell
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGBA"
    scn.view_settings.view_transform = "Standard"
    cam_data = bpy.data.cameras.new("slu_cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = cell / ppm
    cam = bpy.data.objects.new("slu_cam", cam_data)
    scn.collection.objects.link(cam)
    scn.camera = cam
    side = 1 if facing == "right" else -1
    cam.rotation_euler = (math.radians(90), 0, math.radians(90 * side))
    return cam


def render_clip(arm, cam, frames, follow_bone, ppm, cell, facing, tmp: Path):
    scn = bpy.context.scene
    side = 1 if facing == "right" else -1
    pb = arm.pose.bones[follow_bone]
    scn.frame_set(frames[0])
    start = (arm.matrix_world @ pb.head).copy()
    out = []
    for k, f in enumerate(frames):
        scn.frame_set(f)
        hp = arm.matrix_world @ pb.head
        # camera stays at a fixed height with the floor ~8% above the bottom edge; follows ground motion
        cam.location = (hp.x + side * 10, hp.y, (cell / ppm) * (0.5 - 0.08))
        scn.render.filepath = str(tmp / f"f{k:04d}.png")
        bpy.ops.render.render(write_still=True)
        # ground-plane root motion projected to screen x (pixels), relative to first frame
        dx = (hp.y - start.y) * ppm * (1 if facing == "right" else -1)
        out.append({"png": scn.render.filepath, "root_dx": round(dx, 2), "root_px": [cell / 2, cell * 0.92]})
    return out


def pack(cells, sheet_path: Path, pad=2, max_w=2048):
    imgs = []
    for c in cells:
        im = bpy.data.images.load(c["png"])
        w, h = im.size
        a = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)   # bottom-up rows
        bpy.data.images.remove(im)
        ys, xs = np.where(a[..., 3] > 0.01)
        if len(xs) == 0:
            x0 = y0 = 0; x1 = y1 = 1
        else:
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        imgs.append((a[y0:y1, x0:x1], (x0, h - y1), (w, h)))          # offset in top-down coords
    # shelf packing (top-down)
    x = y = shelf = 0
    rects = []
    for crop, _, _ in imgs:
        h, w = crop.shape[:2]
        if x + w + pad > max_w:
            x, y, shelf = 0, y + shelf + pad, 0
        rects.append((x, y, w, h))
        x += w + pad
        shelf = max(shelf, h)
    W = max(r[0] + r[2] for r in rects)
    H = y + shelf
    W2, H2 = 1 << (W - 1).bit_length(), 1 << (H - 1).bit_length()
    sheet = np.zeros((H2, W2, 4), np.float32)
    for (crop, _, _), (rx, ry, w, h) in zip(imgs, rects):
        sheet[H2 - ry - h: H2 - ry, rx: rx + w] = crop                  # back to bottom-up
    img = bpy.data.images.new(sheet_path.stem, W2, H2, alpha=True)
    img.pixels = sheet.ravel()
    img.filepath_raw = str(sheet_path)
    img.file_format = "PNG"
    img.save()
    frames = []
    for (crop, off, size), r, c in zip(imgs, rects, cells):
        frames.append({"rect": list(map(int, r)), "offset": list(map(int, off)), "source_size": list(size),
                       "pivot": [round(c["root_px"][0] - off[0], 1), round(c["root_px"][1] - off[1], 1)],
                       "root_dx": c["root_dx"]})
    return frames, (W2, H2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--game", default="kairos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--cell", type=int)
    a = ap.parse_args(argv())
    cfg = game_cfg(a.game)
    sp = cfg.get("sprites", {})
    cell, ppm = a.cell or sp.get("cell_px", 256), sp.get("pixels_per_metre", 120)
    if a.cell:
        ppm = ppm * a.cell / sp.get("cell_px", 256)
    step, facing = sp.get("frame_step", 2), sp.get("facing", "right")
    model = a.model or sp.get("model")
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for clip in map(Path, a.clips):
        reset()
        meta = json.loads(clip.with_suffix(".clip.json").read_text()) if clip.with_suffix(".clip.json").exists() else {}
        move = clip.stem.split("__")[1] if "__" in clip.stem else clip.stem
        src = import_bvh(clip)
        if model and Path(model).expanduser().exists():
            tgt = import_model(model)
            rig = load_yaml(REPO / f"config/rigs/{sp.get('rig', 'meshy')}.yaml")
            cmap = rig_map(rig, [b.name for b in tgt.data.bones])
            RT.bake(src, tgt, cmap, move)
            arm, follow = tgt, cmap["Hips"]
            meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == tgt or
                      (o.type == "MESH" and o.parent == tgt)]
            src.hide_render = True
        else:
            arm, follow = src, "Hips"
            meshes = mannequin(src)
        if sp.get("toon", True):
            toonify(meshes)
        if sp.get("outline_px", 2):
            outline(meshes, sp.get("outline_px", 2) / ppm)
        cam = setup_scene(cell, ppm, facing, a.engine)
        act = src.animation_data.action
        f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
        frames = list(range(f0, f1 + 1, step))
        with tempfile.TemporaryDirectory() as td:
            cells = render_clip(arm, cam, frames, follow, ppm, cell, facing, Path(td))
            sheet = out / f"{a.game}_{move}.png"
            fr, size = pack(cells, sheet)
        hits = [h // step for h in meta.get("hit_frames", [])]
        atlas = {"meta": {"app": "SLU Mocap", "format": "slu-sprite-atlas", "version": 1, "image": sheet.name,
                          "size": list(size), "cell": cell, "pixels_per_metre": ppm, "facing": facing,
                          "source": clip.name},
                 "animations": {move: {"frames": list(range(len(fr))), "fps": 60 / step,
                                       "loop": bool(meta.get("loop")), "events": {"hit": hits},
                                       "hit_frame": hits[0] if hits else None}},
                 "frames": [{"name": f"{move}_{i:03d}", **f} for i, f in enumerate(fr)]}
        (out / f"{a.game}_{move}.json").write_text(json.dumps(atlas, indent=1))
        index.append({"move": move, "image": sheet.name, "atlas": f"{a.game}_{move}.json", "frames": len(fr)})
        print(f"SPRITES {move}: {len(fr)} frames -> {sheet.name} {size}")
    idx = out / f"{a.game}_sprites_index.json"
    prev = json.loads(idx.read_text()) if idx.exists() else []
    prev = [p for p in prev if p["move"] not in {i["move"] for i in index}] + index
    idx.write_text(json.dumps(sorted(prev, key=lambda p: p["move"]), indent=1))
    print("SPRITES_OK")


if __name__ == "__main__":
    main()
