"""SLU Review - local web app over Google Drive for desktop's /SLU_Mocap/out.

    python3 mac/slu_review.py            (or double-click mac/SLU Review.command)

Grid of takes per game/move with contact-sheet playback + QC badges.
Approve / Retake / Best: Best runs retarget (+ sprites for games with `sprites:`) and copies
results into the game's configured target folders. State lives in out/_review.json.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, threading, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.common import game_config, read_json, write_json  # noqa
from pipeline.export import find_blender  # noqa

CANDIDATES = [Path.home() / "Library/CloudStorage", Path.home() / "Google Drive", Path("/Volumes/GoogleDrive")]


def find_root() -> Path:
    if os.environ.get("SLU_MOCAP_ROOT"):
        return Path(os.environ["SLU_MOCAP_ROOT"]).expanduser()
    for c in CANDIDATES:
        if c.exists():
            for p in c.glob("**/SLU_Mocap"):
                if (p / "out").exists():
                    return p
    sys.exit("Could not find SLU_Mocap on Google Drive. Set SLU_MOCAP_ROOT=/path/to/SLU_Mocap")


ROOT = find_root()
OUT = ROOT / "out"
STATE = OUT / "_review.json"
JOBS: dict[str, str] = {}


def takes():
    st = read_json(STATE, {}) or {}
    rows = []
    for p in sorted(OUT.glob("*/*/*.clip.json")):
        m = read_json(p)
        game, move = p.parent.parent.name, p.parent.name
        rows.append({"id": m["take"], "game": game, "move": move, "qc": m.get("qc", {}),
                     "frames": m["frames"], "loop": m.get("loop"), "hits": m.get("hit_frames", []),
                     "video": f"/files/{game}/{move}/{m['take']}.contact.mp4",
                     "status": st.get(m["take"], {}).get("status", "new"), "job": JOBS.get(m["take"])})
    return rows


def deliver(take_id: str):
    game, move, _ = take_id.split("__")
    d = OUT / game / move
    bvh = d / f"{take_id}.bvh"
    cfg = game_config(game)
    blender = find_blender()
    if not blender:
        JOBS[take_id] = "error: Blender not found (install from blender.org)"
        return
    deliv = OUT / "_delivered" / game
    deliv.mkdir(parents=True, exist_ok=True)
    tmp = d / "_bvh"
    tmp.mkdir(exist_ok=True)
    clean = tmp / f"{game}__{move}.bvh"            # deliver under the move name
    shutil.copy2(bvh, clean)
    shutil.copy2(d / f"{take_id}.clip.json", clean.with_suffix(".clip.json"))
    best = [r for r in takes() if r["game"] == game and r["status"] == "best"]
    try:
        for rig in cfg.get("rigs", ["mixamo"]):
            JOBS[take_id] = f"retargeting ({rig})"
            all_best = []
            for r in best:
                b = OUT / r["game"] / r["move"] / "_bvh" / f"{r['game']}__{r['move']}.bvh"
                if b.exists():
                    all_best.append(str(b))
            subprocess.run([blender, "-b", "--factory-startup", "-P", str(REPO / "blender/retarget.py"), "--",
                            "--rig", rig, "--out", str(deliv / rig), "--batch", f"{game}_anims"] + all_best,
                           check=True, capture_output=True)
        if cfg.get("sprites"):
            JOBS[take_id] = "rendering sprites"
            subprocess.run([blender, "-b", "--factory-startup", "-P", str(REPO / "blender/render_sprites.py"),
                            "--", "--game", game, "--out", str(deliv / "sprites"), str(clean)],
                           check=True, capture_output=True)
        copied = copy_to_targets(game, deliv, cfg)
        JOBS[take_id] = f"delivered -> {', '.join(copied) or '_delivered/'}"
    except subprocess.CalledProcessError as e:
        JOBS[take_id] = "error: " + (e.stderr or b"").decode()[-300:]


def copy_to_targets(game, deliv: Path, cfg) -> list[str]:
    done = []
    t = cfg.get("targets") or {}
    plan = {"godot": ("*.glb",), "unreal": ("*.fbx",), "sprites": ("sprites/*.png", "sprites/*.json")}
    for key, pats in plan.items():
        dst = t.get(key)
        if not dst:
            continue
        dst = Path(dst).expanduser()
        if not dst.parent.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for pat in pats:
            for f in deliv.rglob(pat):
                shutil.copy2(f, dst / f.name)
        done.append(key)
    return done


class H(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path).path
        if u == "/":
            b = (REPO / "mac/review.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(b)
        elif u == "/api/takes":
            self._json({"root": str(ROOT), "takes": takes()})
        elif u.startswith("/files/"):
            p = (OUT / unquote(u[7:])).resolve()
            if OUT.resolve() not in p.parents or not p.exists():
                return self.send_error(404)
            self.path = "/" + str(p.relative_to(OUT.resolve()))
            self.directory = str(OUT.resolve())
            return super().do_GET()
        else:
            self.send_error(404)

    def translate_path(self, path):
        return str(Path(self.directory) / unquote(urlparse(path).path).lstrip("/"))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        tid, status = body.get("id"), body.get("status")
        if status not in ("approved", "retake", "best", "new") or "__" not in (tid or ""):
            return self._json({"error": "bad request"}, 400)
        st = read_json(STATE, {}) or {}
        if status == "best":     # one best per game/move
            key = tid.rsplit("__", 1)[0]
            for k, v in st.items():
                if k.startswith(key + "__") and v.get("status") == "best":
                    v["status"] = "approved"
        st[tid] = {"status": status}
        write_json(STATE, st)
        if status == "best":
            JOBS[tid] = "queued"
            threading.Thread(target=deliver, args=(tid,), daemon=True).start()
        self._json({"ok": True})


def main():
    port = int(os.environ.get("SLU_REVIEW_PORT", 8765))
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"SLU Review on http://127.0.0.1:{port}  ({ROOT})")
    if not os.environ.get("SLU_NO_BROWSER"):
        threading.Timer(0.4, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
