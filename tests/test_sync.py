import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ingest import chirp_template, find_chirp, gcc_phat, AUDIO_SR as SR, match_move
from pipeline.common import load_manifests


def test_chirp_and_clap_sync():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, SR * 6); b = rng.normal(0, 0.05, SR * 6)
    tpl = chirp_template()
    ta, tb = 1.0, 1.4373                     # chirp heard 437.3 ms later in b
    a[int(ta * SR):int(ta * SR) + tpl.size] += tpl; b[int(tb * SR):int(tb * SR) + tpl.size] += 0.6 * tpl
    (ca, sa), (cb, sb) = find_chirp(a), find_chirp(b)
    assert abs((cb - ca) - (tb - ta)) < 1 / SR * 2 and min(sa, sb) > 20
    clap = np.zeros(SR // 50); clap[:40] = rng.normal(0, 1, 40)
    a[int(3 * SR):int(3 * SR) + clap.size] += clap; b[int(3.4373 * SR):int(3.4373 * SR) + clap.size] += clap
    d, pk = gcc_phat(a, b)
    assert abs(d - 0.4373) < 1e-3, d       # positive = event later in b
    # < 1 frame at 60 fps
    assert abs(d - 0.4373) < 1 / 60


def test_whisper_move_matching():
    m = load_manifests()
    assert match_move("okay this is the rifle run loop take two", m) == ("no_extraction", "rifle_run_loop")
    assert match_move("kick round", m) == ("kairos", "kick_round")
    assert match_move("banana", m) == (None, None)


if __name__ == "__main__":
    test_chirp_and_clap_sync(); test_whisper_move_matching(); print("OK")
