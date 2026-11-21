# SLU Mocap

Free, commercially clean, two-iPhone markerless mocap. **Shoot → tap → approve.**

```
SLU Capture (16e front + 11 side) ─▶ Drive/SLU_Mocap/inbox ─▶ Colab: Run all ─▶ Drive/SLU_Mocap/out ─▶ SLU Review (Mac) ─▶ Godot / Unreal / KAIROS sprites
```

## 1 · Setup (once)
1. **Mac:** install [Google Drive for desktop](https://www.google.com/drive/download/) and [Blender](https://www.blender.org/download/). Create `My Drive/SLU_Mocap/`.
2. **Phones:** open `ios/SLUCapture/SLUCapture.xcodeproj` → Signing: pick your Team → Run on each iPhone. Install the Google Drive app on both.
3. In SLU Capture ⚙︎: set **Leader** (16e, front) / **Follower** (11, side), then **Choose Drive › SLU_Mocap › inbox**.
4. Print `calib/charuco_board.pdf` at **100 %** (check the 100 mm bar), tape it to card.

## 2 · Calibrate (each time the tripods move)
Tripods ~3–4 m from the action, ~90° apart, chest height. Leader ⚙︎ → **Calibration mode** → Record → sweep the board for 20 s where both phones see it.

## 3 · Shoot
Tap the subject on each phone to **lock** exposure/focus. Pick game + move on the Leader, **Record** (both phones start together; a chirp marks sync). Start every take in a 1 s **T-pose**. **Retake** marks the last take bad. Keep the skeleton out of the red.

## 4 · Run
Open [`notebooks/SLU_Mocap.ipynb`](notebooks/SLU_Mocap.ipynb) in Colab (T4 GPU) → **Run all**. Tick `WATCH` to process uploads as they land. Open `out/qc_report.html` for the summary.

## 5 · Approve
Double-click `mac/SLU Review.command`. For each move: watch the contact sheet, then **★ Best take**. That retargets to the game's rigs and copies files into the paths in `config/games/<game>.yaml` (`targets:`): Godot GLB (one per game with every move as a named action), Unreal FBX, and KAIROS sprite sheets + JSON atlas.

---
**Add moves:** edit `manifests/*.yaml`, run `python3 manifests/build_json.py`, push (phones refresh on launch).
**Old clips (no app):** put `front/` + `side/` folders on Drive, set `LEGACY_FOLDER` in the notebook (clap sync + Whisper naming, unmatched → `review/`).
**Rokoko / other BVH-FBX:** `blender -b -P blender/retarget.py -- --rig mixamo --out OUT clip.fbx`.
**Face:** Live Link Face (ARKit mode), slate = take id → CSV into `inbox/face/` → `out/…/<take>.face.json`.
**Tests:** `for t in tests/test_*.py; do python3 $t; done` · Licenses: [LICENSES.md](LICENSES.md)
