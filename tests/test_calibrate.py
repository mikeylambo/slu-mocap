"""Synthetic ChArUco calibration: render the board into two virtual phones, recover the rig."""
import sys, tempfile
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import calibrate as C
from tests.synth import look_at, relative_to
from scipy.spatial.transform import Rotation as R


def render_video(path, cam, poses, board_img, ppm):
    w, h = cam["size"]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    bh, bw = board_img.shape
    for Rb, tb in poses:
        # board pixel (u,v) -> board metres (u/ppm, v/ppm, 0) -> world -> camera image
        corners_px = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], float)
        Xb = np.c_[corners_px / ppm, np.zeros(4)]
        Xw = Xb @ Rb.T + tb
        xc = Xw @ cam["R"].T + cam["t"]
        x = xc @ cam["K"].T
        x = (x[:, :2] / x[:, 2:]).astype(np.float32)
        H = cv2.getPerspectiveTransform(corners_px.astype(np.float32), x)
        img = cv2.warpPerspective(board_img, H, (w, h), borderValue=110, flags=cv2.INTER_AREA)
        wr.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    wr.release()


def test_stereo_calibration():
    b = C.make_board()
    ppm = 4000
    sx, sy = C.BOARD["squares_x"] * C.BOARD["square_m"], C.BOARD["squares_y"] * C.BOARD["square_m"]
    img = b.generateImage((int(sx * ppm), int(sy * ppm)), marginSize=0, borderBits=1)
    tgt = np.array([0, 1.2, 1.6])
    front = look_at([0, 1.2, 0.0], tgt, f=1400, size=(1280, 720))
    side = look_at([1.3, 1.2, 1.0], tgt, f=1500, size=(1280, 720))
    rng = np.random.default_rng(3)
    poses = []
    for k in range(60):
        m = np.array([0.65, 1.2, 0.5])            # between the two phones
        z = (tgt - m) / np.linalg.norm(tgt - m)    # board z points away from the viewers
        x = np.cross([0, -1, 0], z); x /= np.linalg.norm(x)
        Rb = np.stack([x, np.cross(z, x), z], 1)
        Rb = R.from_rotvec(rng.uniform(-.45, .45, 3)).as_matrix() @ Rb
        tb = tgt + rng.uniform(-0.25, 0.25, 3) - Rb @ np.array([sx / 2, sy / 2, 0])
        poses.append((Rb, tb))
    with tempfile.TemporaryDirectory() as d:
        vids = {"leader": Path(d) / "l.mp4", "follower": Path(d) / "f.mp4"}
        render_video(vids["leader"], front, poses, img, ppm)
        render_video(vids["follower"], side, poses, img, ppm)
        samples, sizes = C.collect(vids, {"leader": 0, "follower": 0}, "leader", step_s=1 / 30)
        rep = C.calibrate_views(samples, sizes, "leader")
    rel = relative_to(front, side)
    t_est = np.array(rep["cams"]["follower"]["t"])
    R_est = np.array(rep["cams"]["follower"]["R"])
    ang = np.degrees(np.linalg.norm(R.from_matrix(R_est @ rel["R"].T).as_rotvec()))
    print("rms", rep["worst_rms_px"], "baseline err mm", np.linalg.norm(t_est - rel["t"]) * 1000, "rot err deg", ang)
    assert rep["ok"] and np.linalg.norm(t_est - rel["t"]) < 0.03 and ang < 0.5


if __name__ == "__main__":
    test_stereo_calibration(); print("OK")
