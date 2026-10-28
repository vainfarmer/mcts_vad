"""Agent C gate check: run the BUE spec arm at Lmax=40 (the PROPOSED setting) and
Lmax=60 (the pre-registered setting) across EVERY available full cache, not just
the 3 the proposal used. Tests for cache selection.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'video_anomaly_detection')); sys.path.insert(0, str(HERE))
import analysis_v142_bue_verify as BV

ARMS = {
    'BUE L=40 (proposed)': dict(alpha=1.0, gate=True, union=True, Lmax=40.0),
    'BUE L=60 (prereg)':   dict(alpha=1.0, gate=True, union=True, Lmax=60.0),
}

if __name__ == '__main__':
    dirs = [Path(x) for x in sys.argv[1:]]
    agg = {}
    for d in dirs:
        for k, v in BV.run(d, ARMS).items():
            agg.setdefault(k, []).append(v)
    import pickle
    BV.CACHE.write_bytes(pickle.dumps(BV._CV))
    print('\n=== MEANS over %d caches ===' % len(dirs))
    for k, v in agg.items():
        print(f'  {k:24s} mean={np.mean(v):+.5f} n+={sum(1 for x in v if x>0)}/{len(v)} '
              f'{[round(x,4) for x in v]}')
