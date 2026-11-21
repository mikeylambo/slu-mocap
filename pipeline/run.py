"""SLU Mocap orchestrator: inbox -> out, incremental. Called by the Colab notebook.

    python -m pipeline.run --root /content/drive/MyDrive/SLU_Mocap [--watch] [--force] [--only ID]
"""
from __future__ import annotations
import argparse, time, traceback
from pathlib import Path
import numpy as np
from . import skeleton as S, ingest, calibrate, keypoints, triangulate as TR, cleanup, export, qc, face
from .common import (Take, PipelineError, code_hash, file_hash, game_config, log, move_info,
                     read_json, root_dirs, write_json)
from . import solve as SV


def calib_for(take: Take, calib_dir: Path) -> Path | None:
    """Calibration with the same setup_id as the take, else the latest one recorded before it."""
    setup = next((v["meta"].get("setupId") for v in take.views.values() if v["meta"].get("setupId")), None)
    cands = sorted(calib_dir.glob("*.calib.json"), key=lambda p: p.stat().st_mtime)
    if setup:
        same = [p for p in cands if (read_json(p) or {}).get("setupId") == setup]
        if same:
            return same[-1]
    return cands[-1] if cands else None


def aux_vectors(X):
    lr = X[:, S.LEAR] - X[:, S.REAR]
    fwd = X[:, S.NOSE] - 0.5 * (X[:, S.LEAR] + X[:, S.REAR])
    return {"head_x": lr, "head_fwd": fwd}


def process_take(take: Take, d: dict, force=False, device="cuda") -> dict | None:
    cfg = game_config(take.game)
    info = move_info(take.game, take.move)
    vids = [v["video"] for v in take.views.values()]
    fp = file_hash(vids, code_hash())
    out_dir = d["out"] / take.game / take.move
    state_p = d["cache"] / take.id / "state.json"
    if not force and (read_json(state_p) or {}).get("hash") == fp and (out_dir / f"{take.id}.clip.json").exists():
        return None
    log("take", f"{take.id}  views={sorted(take.views)}")
    sync = ingest.sync_take(take)
    ref = sync["ref"]
    if take.is_calibration:
        setup = take.views[ref]["meta"].get("setupId", take.id)
        rep = calibrate.run(take, sync, d["calib"] / f"{take.id}.calib.json")
        rep["setupId"] = setup
        write_json(d["calib"] / f"{take.id}.calib.json", rep)
        write_json(state_p, {"hash": fp})
        return {"calibration": take.id, "rms": rep["worst_rms_px"]}
    # 2D keypoints for every view (cached)
    kp = {r: keypoints.load(keypoints.run_video(v["video"], d["cache"] / take.id / f"{r}.kp.npz", device))
          for r, v in take.views.items()}
    fps = float(kp[ref]["fps"])
    T = kp[ref]["kp"].shape[0]
    times = np.arange(T) / fps
    samples = {r: keypoints.resample(kp[r], times + sync["offsets"][r]) for r in kp}
    # restrict to frames where all views exist
    valid = np.all([s[1].max(1) > 0 for s in samples.values()], 0)
    if valid.sum() > 0.5 * fps:
        a, b = np.where(valid)[0][[0, -1]]
        samples = {r: (s[0][a:b + 1], s[1][a:b + 1]) for r, s in samples.items()}
        frame0 = a
    else:
        frame0 = 0
    single = len(take.views) < 2
    reproj = None
    calp = calib_for(take, d["calib"])
    if not single and calp is None:
        log("take", f"{take.id}: no calibration yet -> single-view fallback (record calibration!)")
        single = True
    if not single:
        cams = calibrate.load_cams(calp)
        missing = [r for r in samples if r not in cams]
        if missing:
            raise PipelineError(f"{take.id}: calibration {calp.name} has no camera for {missing}")
        X, conf, reproj = TR.triangulate(cams, samples)
    else:
        v3 = d["inbox"].rglob(f"{take.id}__{ref}.vision3d.json")
        v3 = next(v3, None)
        meta = take.views[ref]["meta"]
        if v3:
            X, conf = TR.from_vision3d(read_json(v3)["frames"], len(samples[ref][0]), fps,
                                       start_s=frame0 / fps)
        else:
            cam = _guess_cam(meta, kp[ref])
            X, conf = TR.lift_single_view(*samples[ref], cam)
    conf_raw = conf.copy()
    X, conf, filled = TR.fill_gaps(X, conf, fps)
    X, _ = TR.world_align(X, conf)
    Pj, W = S.joints_from_keypoints(X, conf)
    clip, q = cleanup.run(Pj, W, fps, cfg, take.move, info, aux_vectors(X))
    clip["src_index"] = clip["src_index"] + frame0
    q.update(sync_method=sync["method"], chirp_check_ms=sync["chirp_check_ms"])
    Pprev, _ = SV.fk(clip.get("root_track", clip["root"]), clip["quats"], clip["lengths"], clip["offsets"])
    q = qc.metrics(Pprev, conf_raw, reproj, single, q, cfg)
    meta = export.run(clip, out_dir, take.id, cfg, info, q)
    qc.contact_sheet(take.views[ref]["video"], clip["src_index"], Pprev, out_dir / f"{take.id}.contact.mp4",
                     fps=fps, hits=set(clip["hit_frames"]), contacts=clip["contacts"],
                     title=take.id)
    write_json(state_p, {"hash": fp})
    log("take", f"{take.id}: {q['grade'].upper()}  slide {q['foot_slide_cm']:.1f}cm  "
                f"jitter {q['jitter_mm']:.1f}mm  frames {meta['frames']}")
    return meta


def _guess_cam(meta, kp):
    w, h = map(int, kp["size"])
    f = meta.get("focalPx") or 0.82 * max(w, h)       # iPhone 1x wide ~ 26mm-equiv
    return {"K": np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), "dist": np.zeros(5),
            "R": np.eye(3), "t": np.zeros(3), "size": [w, h]}


def run_all(root, force=False, only=None, device="cuda") -> int:
    d = root_dirs(root)
    takes = ingest.discover(d["inbox"])
    # calibrations first so later takes can use them
    takes.sort(key=lambda t: (not t.is_calibration, t.id))
    done, failed = 0, 0
    for t in takes:
        if only and only not in t.id:
            continue
        try:
            if process_take(t, d, force, device) is not None:
                done += 1
        except PipelineError as e:
            failed += 1
            log("ERROR", str(e))
            write_json(d["out"] / "_errors" / f"{t.id}.json", {"error": str(e)})
        except Exception:
            failed += 1
            log("ERROR", f"{t.id}\n{traceback.format_exc()}")
            write_json(d["out"] / "_errors" / f"{t.id}.json", {"error": traceback.format_exc()})
    face.run(d["inbox"], d["out"])
    qc.report(d["out"])
    log("done", f"{done} processed, {failed} failed, {len(takes)} takes in inbox -> {d['out']}/qc_report.html")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--watch", action="store_true", help="poll the inbox every 30 s")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    while True:
        run_all(a.root, a.force, a.only, a.device)
        if not a.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
