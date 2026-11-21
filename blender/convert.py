"""blender -b -P blender/convert.py -- in.bvh out.fbx   (canonical BVH -> FBX, 60 fps)"""
import sys
import bpy

a = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 60
bpy.ops.import_anim.bvh(filepath=a[0], global_scale=0.01, frame_start=1, use_fps_scale=False,
                        update_scene_fps=True, update_scene_duration=True, rotate_mode="NATIVE")
bpy.ops.export_scene.fbx(filepath=a[1], use_selection=False, add_leaf_bones=False,
                         bake_anim=True, bake_anim_use_all_actions=False, bake_anim_simplify_factor=0.0,
                         object_types={"ARMATURE"})
