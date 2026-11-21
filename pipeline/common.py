"""Shared helpers: paths, config, manifests, take naming, caching, logging."""
from __future__ import annotations
import hashlib, json, os, re, sys, time
from dataclasses import dataclass, field
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]
TAKE_RE = re.compile(r"^(?P<game>[a-z0-9_]+)__(?P<move>[a-z0-9_]+)__t(?P<take>\d{3})__(?P<role>leader|follower|cam\d)$")
VIDEO_EXT = {".mov", ".mp4", ".m4v"}


def log(stage: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {stage:<11} {msg}", flush=True)


class PipelineError(RuntimeError):
    pass


def load_yaml(p) -> dict:
    with open(p) as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def game_config(game: str) -> dict:
    base = load_yaml(REPO / "config/games/_defaults.yaml")
    p = REPO / f"config/games/{game}.yaml"
    return deep_merge(base, load_yaml(p) if p.exists() else {})


def load_manifests() -> dict[str, dict[str, dict]]:
    """{game: {move: fields}}; shared moves are merged into every game."""
    shared = (load_yaml(REPO / "manifests/shared.yaml").get("moves") or {})
    out = {}
    for p in sorted((REPO / "manifests").glob("*.yaml")):
        if p.stem == "shared":
            continue
        m = load_yaml(p)
        out[p.stem] = {**shared, **(m.get("moves") or {})}
    return out


def move_info(game: str, move: str) -> dict:
    info = dict(load_manifests().get(game, {}).get(move) or {})
    info.setdefault("loop", move.endswith("_loop"))
    info.setdefault("tags", [])
    return info


@dataclass
class Take:
    game: str
    move: str
    take: int
    views: dict = field(default_factory=dict)   # role -> {"video": Path, "meta": dict}

    @property
    def id(self) -> str:
        return f"{self.game}__{self.move}__t{self.take:03d}"

    @property
    def is_calibration(self) -> bool:
        return self.game == "_calib"


def file_hash(paths, extra: str = "") -> str:
    """Cheap content fingerprint (size + mtime + first/last MB) for incremental runs."""
    h = hashlib.sha1(extra.encode())
    for p in sorted(map(Path, paths)):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}".encode())
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
            if st.st_size > 2 << 20:
                f.seek(-(1 << 20), os.SEEK_END)
                h.update(f.read())
    return h.hexdigest()[:16]


def code_hash() -> str:
    h = hashlib.sha1()
    for p in sorted((REPO / "pipeline").glob("*.py")) + sorted((REPO / "config").rglob("*.yaml")):
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    tmp.replace(p)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def root_dirs(root) -> dict[str, Path]:
    root = Path(root)
    d = {k: root / k for k in ("inbox", "out", "review", "cache", "calib", "models")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d
