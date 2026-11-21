"""ingest: discover takes in the inbox, pair views, compute sub-frame sync offsets.

Sync priority: device sharedStart metadata (MultipeerConnectivity clock) -> start chirp
matched filter -> audio GCC-PHAT (clap / legacy). The chirp is always cross-checked when
present and reported in QC.

    python -m pipeline.ingest --root /content/drive/MyDrive/SLU_Mocap [--legacy]
"""
from __future__ import annotations
import argparse, shutil, subprocess
from pathlib import Path
import numpy as np
from .common import (TAKE_RE, VIDEO_EXT, Take, PipelineError, log, read_json, write_json,
                     load_manifests, root_dirs)

AUDIO_SR = 16000
CHIRP = dict(f0=2000.0, f1=6000.0, dur=0.08)   # must match ios ChirpPlayer


# ---- discovery ------------------------------------------------------------------------
def discover(inbox: Path) -> list[Take]:
    takes: dict[str, Take] = {}
    for p in sorted(inbox.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT or p.name.startswith("."):
            continue
        m = TAKE_RE.match(p.stem)
        if not m:
            continue
        t = Take(m["game"], m["move"], int(m["take"]))
        t = takes.setdefault(t.id, t)
        meta = read_json(p.with_suffix(".json"), {}) or {}
        t.views[m["role"]] = {"video": p, "meta": meta}
    # takes marked bad by Retake on the phone are skipped
    return [t for t in takes.values() if not any(v["meta"].get("bad") for v in t.views.values())]


# ---- video / audio probing -------------------------------------------------------------
def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True)
    import json
    s = json.loads(out.stdout)["streams"][0]
    num, den = map(float, s["r_frame_rate"].split("/"))
    return {"width": s["width"], "height": s["height"], "fps": num / den,
            "frames": int(s.get("nb_read_packets") or 0)}


def load_audio(video: Path, seconds: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(video)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-ac", "1", "-ar", str(AUDIO_SR), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, np.float32).astype(np.float64)


def chirp_template(sr=AUDIO_SR) -> np.ndarray:
    t = np.arange(int(CHIRP["dur"] * sr)) / sr
    k = (CHIRP["f1"] - CHIRP["f0"]) / CHIRP["dur"]
    return np.sin(2 * np.pi * (CHIRP["f0"] * t + 0.5 * k * t * t)) * np.hanning(t.size)


def find_chirp(audio: np.ndarray, sr=AUDIO_SR):
    """Return (time_s, snr) of the strongest chirp onset."""
    if audio.size < sr // 4:
        return None, 0.0
    tpl = chirp_template(sr)
    n = 1 << int(np.ceil(np.log2(audio.size + tpl.size)))
    corr = np.fft.irfft(np.fft.rfft(audio, n) * np.conj(np.fft.rfft(tpl, n)), n)[: audio.size]
    env = np.abs(corr)
    i = int(np.argmax(env))
    snr = env[i] / (np.median(env) + 1e-12)
    return i / sr, float(snr)


def gcc_phat(a: np.ndarray, b: np.ndarray, sr=AUDIO_SR, max_shift_s=10.0) -> tuple[float, float]:
    """Delay of b relative to a in seconds (positive: event happens later in b)."""
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    R = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n)
    m = int(max_shift_s * sr)
    cc = np.concatenate((cc[-m:], cc[: m + 1]))
    i = int(np.argmax(np.abs(cc)))
    peak = np.abs(cc[i]) / (np.mean(np.abs(cc)) + 1e-12)
    # parabolic sub-sample refinement
    if 0 < i < cc.size - 1:
        y0, y1, y2 = np.abs(cc[i - 1: i + 2])
        i = i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return (i - m) / sr, float(peak)


# ---- sync --------------------------------------------------------------------------------
def sync_take(take: Take) -> dict:
    """Return {"ref": role, "offsets": {role: seconds}, "method", "chirp_check_ms", ...}.

    offsets[role] = time (s) into that video at which the reference video's t=0 happens.
    So frame f of reference aligns with time f/fps + offsets[role] in the other video.
    """
    roles = sorted(take.views, key=lambda r: (r != "leader", r))
    ref = roles[0]
    res = {"ref": ref, "offsets": {ref: 0.0}, "method": "single", "chirp_check_ms": None}
    if len(roles) == 1:
        return res
    rm = take.views[ref]["meta"]
    a_ref = None
    for r in roles[1:]:
        m = take.views[r]["meta"]
        off, method = None, None
        if "sharedStart" in rm and "sharedStart" in m:
            # sharedStart = shared-clock time of first video frame
            off, method = float(rm["sharedStart"]) - float(m["sharedStart"]), "clock"
        a_ref = load_audio(take.views[ref]["video"], 30) if a_ref is None else a_ref
        a = load_audio(take.views[r]["video"], 30)
        c0, s0 = find_chirp(a_ref)
        c1, s1 = find_chirp(a)
        chirp_off = (c1 - c0) if (c0 is not None and c1 is not None and min(s0, s1) > 20) else None
        if off is None and chirp_off is not None:
            off, method = chirp_off, "chirp"
        if off is None and a_ref.size and a.size:
            off, pk = gcc_phat(a_ref, a)
            method = f"audio_gcc(peak={pk:.1f})"
            if pk < 8:
                log("ingest", f"{take.id}: weak audio sync peak {pk:.1f} - check clap")
        if off is None:
            raise PipelineError(f"{take.id}: no way to sync {r} (no metadata, no audio)")
        res["offsets"][r] = off
        res["method"] = method
        if chirp_off is not None:
            res["chirp_check_ms"] = round((chirp_off - off) * 1000, 2)
    return res


# ---- legacy footage (plain camera clips with a clap + spoken move name) -------------------
def ingest_legacy(src: Path, inbox: Path, review: Path, game_hint: str | None = None):
    """Pair clips recorded without the app and name them from speech.

    Expect src/front/*.mov and src/side/*.mov (or any two subfolders). Clips are paired by
    audio correlation; Whisper (small, MIT) transcribes the first seconds to find the move.
    """
    try:
        import whisper  # openai-whisper, MIT
    except ImportError:
        raise PipelineError("pip install openai-whisper  (needed for --legacy naming)")
    model = whisper.load_model("small")
    manifests = load_manifests()
    dirs = [d for d in sorted(src.iterdir()) if d.is_dir()]
    if len(dirs) < 2:
        dirs = [src]
    clips = {d.name: sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXT) for d in dirs}
    names = list(clips)
    lead = clips[names[0]]
    others = {n: list(clips[n]) for n in names[1:]}
    counters: dict[str, int] = {}
    for clip in lead:
        a = load_audio(clip, 20)
        group = {"leader": clip}
        for k, n in enumerate(names[1:]):
            best = max(others[n], key=lambda c: gcc_phat(a, load_audio(c, 20))[1], default=None)
            if best is not None:
                group["follower" if k == 0 else f"cam{k+1}"] = best
                others[n].remove(best)
        text = model.transcribe(str(clip), language="en", fp16=False,
                                clip_timestamps=[0, 12])["text"].lower()
        game, move = match_move(text, manifests, game_hint)
        if not move:
            dst = review / clip.stem
            dst.mkdir(parents=True, exist_ok=True)
            for role, p in group.items():
                shutil.copy2(p, dst / f"{role}{p.suffix.lower()}")
            write_json(dst / "transcript.json", {"text": text})
            log("legacy", f"{clip.name}: no move matched in '{text.strip()[:60]}' -> review/")
            continue
        key = f"{game}__{move}"
        n = counters[key] = max(counters.get(key, 0), _max_take(inbox, key)) + 1
        for role, p in group.items():
            dst = inbox / "legacy" / f"{key}__t{n:03d}__{role}{p.suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            write_json(dst.with_suffix(".json"), {"game": game, "move": move, "take": n,
                                                  "role": role, "legacy": True, "transcript": text})
        log("legacy", f"{clip.name} -> {key}__t{n:03d} ({len(group)} views)")


def _max_take(inbox: Path, key: str) -> int:
    ts = [int(m["take"]) for p in inbox.rglob(f"{key}__t*") if (m := TAKE_RE.match(p.stem))]
    return max(ts, default=0)


def match_move(text: str, manifests: dict, game_hint=None):
    import difflib, re
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    best = (0.0, None, None)
    for game, moves in manifests.items():
        if game_hint and game != game_hint:
            continue
        for move in moves:
            spoken = move.replace("_loop", "").replace("_", " ")
            if spoken in words:
                score = 1.0 + len(spoken) / 100
            else:
                score = max((difflib.SequenceMatcher(None, spoken, words[i:i + len(spoken)]).ratio()
                             for i in range(0, max(1, len(words) - len(spoken) + 1))), default=0)
            if score > best[0]:
                best = (score, game, move)
    return (best[1], best[2]) if best[0] >= 0.8 else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--legacy", help="folder of app-less clips to import")
    ap.add_argument("--game", help="restrict legacy matching to one game")
    a = ap.parse_args()
    d = root_dirs(a.root)
    if a.legacy:
        ingest_legacy(Path(a.legacy), d["inbox"], d["review"], a.game)
    for t in discover(d["inbox"]):
        print(t.id, sorted(t.views), sync_take(t))


if __name__ == "__main__":
    main()
