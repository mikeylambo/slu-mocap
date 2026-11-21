"""calibrate: ChArUco -> per-phone intrinsics + stereo extrinsics. Fails loudly > 1.5 px.

Board: 7x5 squares, 38 mm squares, 28 mm markers, DICT_5X5_100 (calib/charuco_board.pdf).
Print at 100 % on A4/Letter, tape to something flat, check one square with a ruler.

    python -m pipeline.calibrate --board-pdf calib/charuco_board.pdf     # regenerate board
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from .common import PipelineError, log, write_json, read_json

BOARD = dict(squares_x=7, squares_y=5, square_m=0.038, marker_m=0.028, dict="DICT_5X5_100")
MAX_RMS_PX = 1.5


def make_board(spec=BOARD):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dict"]))
    b = cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]), spec["square_m"], spec["marker_m"], d)
    b.setLegacyPattern(False)
    return b


def board_pdf(path: Path, spec=BOARD, dpi=300):
    from PIL import Image, ImageDraw
    b = make_board(spec)
    mm = lambda v: int(round(v / 25.4 * dpi))
    w_mm, h_mm = spec["squares_x"] * spec["square_m"] * 1000, spec["squares_y"] * spec["square_m"] * 1000
    img = b.generateImage((mm(w_mm), mm(h_mm)), marginSize=0, borderBits=1)
    page = Image.new("L", (mm(297), mm(210)), 255)       # A4 landscape (fits Letter too)
    ox, oy = (page.width - img.shape[1]) // 2, (page.height - img.shape[0]) // 2 - mm(6)
    page.paste(Image.fromarray(img), (ox, oy))
    dr = ImageDraw.Draw(page)
    label = (f"SLU Mocap ChArUco {spec['squares_x']}x{spec['squares_y']}  square {spec['square_m']*1000:.0f} mm  "
             f"marker {spec['marker_m']*1000:.0f} mm  {spec['dict']}  -  PRINT AT 100% (no fit-to-page). "
             f"Check: this bar = 100 mm")
    dr.text((mm(12), page.height - mm(14)), label, fill=0)
    dr.rectangle([mm(12), page.height - mm(8), mm(112), page.height - mm(6)], fill=0)
    page.save(path, "PDF", resolution=dpi)


def detect(gray, board, detector=None):
    detector = detector or cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return None, None
    return ch_corners.reshape(-1, 2), ch_ids.ravel()


def _frames(video, step, start_s=0.0, offset_s=0.0):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, n


def collect(videos: dict, offsets: dict, ref: str, step_s=0.15, board=None):
    """Read synced frames from each view -> per-frame detections {role: (corners, ids)}."""
    board = board or make_board()
    det = cv2.aruco.CharucoDetector(board)
    caps = {r: cv2.VideoCapture(str(v)) for r, v in videos.items()}
    fps = caps[ref].get(cv2.CAP_PROP_FPS) or 60
    n = int(caps[ref].get(cv2.CAP_PROP_FRAME_COUNT))
    sizes, samples = {}, []
    for f in range(0, n, max(1, int(round(step_s * fps)))):
        row = {}
        for r, cap in caps.items():
            fr_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            idx = int(round((f / fps + offsets.get(r, 0.0)) * fr_fps))
            if idx < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sizes[r] = g.shape[::-1]
            c, i = detect(g, board, det)
            if c is not None:
                row[r] = (c, i)
        samples.append(row)
    return samples, sizes


def calibrate_views(samples, sizes, ref: str, board=None, max_frames=80) -> dict:
    board = board or make_board()
    obj_all = board.getChessboardCorners()
    cams = {}
    for r, size in sizes.items():
        dets = [s[r] for s in samples if r in s and len(s[r][1]) >= 10]
        if len(dets) < 8:
            raise PipelineError(f"calibration: board seen in only {len(dets)} frames by {r} (need 8+). "
                                "Wave the board slowly, tilted, filling more of the frame.")
        dets = _spread(dets, max_frames)
        objs = [obj_all[i].astype(np.float32) for _, i in dets]
        imgs = [c.astype(np.float32) for c, _ in dets]
        flags = cv2.CALIB_FIX_K3
        rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None, flags=flags)
        cams[r] = {"K": K, "dist": dist.ravel(), "size": list(size), "rms_intrinsic": float(rms),
                   "R": np.eye(3), "t": np.zeros(3)}
        log("calibrate", f"{r}: intrinsics rms {rms:.3f}px from {len(dets)} frames, f={K[0,0]:.0f}px")
    for r in sizes:
        if r == ref:
            continue
        objs, ia, ib = [], [], []
        for s in samples:
            if ref in s and r in s:
                common, ka, kb = np.intersect1d(s[ref][1], s[r][1], return_indices=True)
                if len(common) >= 8:
                    objs.append(obj_all[common].astype(np.float32))
                    ia.append(s[ref][0][ka].astype(np.float32))
                    ib.append(s[r][0][kb].astype(np.float32))
        if len(objs) < 6:
            raise PipelineError(f"calibration: board seen by both {ref} and {r} in only {len(objs)} "
                                "frames (need 6+). Keep it visible to both phones at once.")
        sel = _spread(list(zip(objs, ia, ib)), 80)
        objs, ia, ib = map(list, zip(*sel))
        # joint refinement: intrinsics from the per-camera pass are the initial guess
        rms, K1, d1, K2, d2, R, t, _, _ = cv2.stereoCalibrate(
            objs, ia, ib, cams[ref]["K"], cams[ref]["dist"], cams[r]["K"], cams[r]["dist"],
            tuple(sizes[ref]), flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-9))
        cams[ref].update(K=K1, dist=d1.ravel())
        cams[r].update(K=K2, dist=d2.ravel(), R=R, t=t.ravel(), rms_stereo=float(rms), stereo_frames=len(objs))
        log("calibrate", f"{ref}<->{r}: stereo rms {rms:.3f}px, baseline {np.linalg.norm(t):.2f} m, "
                         f"{len(objs)} frames")
    worst = max(max(c["rms_intrinsic"], c.get("rms_stereo", 0)) for c in cams.values())
    report = {"ref": ref, "worst_rms_px": worst, "ok": worst <= MAX_RMS_PX, "board": BOARD,
              "cams": {r: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
                       for r, c in cams.items()}}
    if worst > MAX_RMS_PX:
        raise PipelineError(f"calibration reprojection error {worst:.2f}px > {MAX_RMS_PX}px. "
                            "Re-record calibration: steady light, board flat, slower waves, "
                            "fill more of the frame, both phones see it.")
    return report


def _spread(items, k):
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def load_cams(path) -> dict:
    c = read_json(path)
    return {r: {k: np.asarray(v) if isinstance(v, list) else v for k, v in cam.items()}
            for r, cam in c["cams"].items()}


def run(take, sync: dict, out: Path) -> dict:
    vids = {r: v["video"] for r, v in take.views.items()}
    samples, sizes = collect(vids, sync["offsets"], sync["ref"])
    rep = calibrate_views(samples, sizes, sync["ref"])
    write_json(out, rep)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-pdf")
    a = ap.parse_args()
    if a.board_pdf:
        board_pdf(Path(a.board_pdf))
        print("wrote", a.board_pdf)
