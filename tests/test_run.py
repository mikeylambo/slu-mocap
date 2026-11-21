"""Orchestrator integration test: fake inbox (videos + metadata + cached 2D keypoints + calibration)."""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.synth import walk, keypoints_from_joints
from tests.test_pipeline import _two_view
from pipeline import run as RUN
from pipeline.common import write_json, read_json


def test_run_all():
    root, Q, P, G = walk(seconds=3)
    K3 = keypoints_from_joints(P, G)
    cams, kps = _two_view(K3)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inbox = d / "inbox"; inbox.mkdir()
        tid = "no_extraction__walk_loop__t001"
        T = len(P)
        for k, r in enumerate(("leader", "follower")):
            v = inbox / f"{tid}__{r}.mov"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=60",
                            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", str(T / 60), "-shortest",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(v)], check=True)
            write_json(v.with_suffix(".json"), {"sharedStart": 100.0, "fps": 60, "setupId": "s1", "role": r})
            c = d / "cache" / tid; c.mkdir(parents=True, exist_ok=True)
            np.savez(c / f"{r}.kp.npz", kp=kps[r][0], score=kps[r][1], fps=60.0, size=np.array([1920, 1080]))
        calib = d / "calib"; calib.mkdir()
        write_json(calib / "_calib__calib__t001.calib.json",
                   {"setupId": "s1", "worst_rms_px": 0.3,
                    "cams": {r: {k: np.asarray(v).tolist() if not isinstance(v, list) else v for k, v in c.items()}
                             for r, c in cams.items()}})
        assert RUN.run_all(d, device="cpu") == 0
        out = d / "out/no_extraction/walk_loop"
        meta = read_json(out / f"{tid}.clip.json")
        print(meta["qc"])
        assert (out / f"{tid}.bvh").exists() and (out / f"{tid}.contact.mp4").exists()
        assert meta["qc"]["grade"] == "green", meta["qc"]
        assert (d / "out/qc_report.html").exists()
        # incremental: second run does nothing
        assert RUN.run_all(d, device="cpu") == 0


if __name__ == "__main__":
    test_run_all(); print("OK")
