"""Regenerate manifests/manifests.json (read by SLU Capture) from the YAML manifests.
    python3 manifests/build_json.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import load_manifests
m = load_manifests()
out = {"version": 1, "games": [{"id": g, "moves": [{"id": k, "loop": bool(v.get("loop")), "notes": v.get("notes", ""),
       "tags": v.get("tags", [])} for k, v in mv.items()]} for g, mv in m.items()]}
txt = json.dumps(out, indent=1)
(Path(__file__).parent / "manifests.json").write_text(txt)
(Path(__file__).resolve().parents[1] / "ios/SLUCapture/Shared/manifests.json").write_text(txt)   # bundled fallback
print("wrote manifests.json")
