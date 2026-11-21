# Licenses - shipping path audit

**Rule:** nothing that ships in a game (the animation data) may depend on a non-commercial model.
No SMPL / SMPL-X / GVHMR / WHAM / 4D-Humans / HaMeR or anything SMPL-derived is used anywhere.

| Component | Role | License | Commercial use |
|---|---|---|---|
| SLU Mocap code (this repo) | everything | yours | ✅ |
| [rtmlib](https://github.com/Tau-J/rtmlib) | RTMW/YOLOX inference wrapper | Apache-2.0 | ✅ |
| RTMW whole-body weights (MMPose) | 2D keypoints | Apache-2.0 (weights released by OpenMMLab) | ✅ ¹ |
| YOLOX-m (HumanArt) weights | person detection | Apache-2.0 | ✅ ¹ |
| ONNX Runtime | inference | MIT | ✅ |
| OpenCV (contrib) incl. ArUco/ChArUco | calibration, video | Apache-2.0 | ✅ |
| NumPy / SciPy | maths | BSD-3 | ✅ |
| PyYAML | config | MIT | ✅ |
| Pillow | board PDF | MIT-CMU (HPND) | ✅ |
| FFmpeg (CLI, LGPL build) | decode / encode previews | LGPL-2.1+ | ✅ tool only, not shipped |
| Blender (headless) | retarget, FBX/GLB, sprites | GPL-3 | ✅ output is yours; GPL does not cover rendered/exported data |
| Apple Vision 3D body pose | live framing + single-view fallback | Apple SDK (on-device) | ✅ |
| Geometric single-view lifter (`triangulate.lift_single_view`) | fallback | this repo (no learned prior) | ✅ |
| openai-whisper `small` | naming legacy clips only (not in shipped data) | MIT | ✅ |
| Live Link Face (optional face) | ARKit blendshape CSV | Epic, free app; ARKit data is yours | ✅ |

¹ Weights are Apache-2.0. Some public datasets in RTMW's training mix carry research-only
terms; the weights themselves are licensed Apache-2.0 by OpenMMLab and are widely used
commercially. If your publisher wants zero dataset ambiguity, set
`keypoints.run_video(..., mode="lightweight")` or swap the pose ONNX for an RTMPose model
trained on COCO only (CC-BY-4.0 annotations) - the pipeline is model-agnostic.

Not used anywhere: SMPL, SMPL-X, SMPL-H, MANO, GVHMR, WHAM, 4D-Humans, HMR2, HaMeR,
MotionBERT/VideoPose3D (H3.6M-trained), Rokoko Vision (its exports can be *imported* - the
licence of that data is governed by your Rokoko plan).
