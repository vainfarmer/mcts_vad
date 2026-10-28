"""v203 CBT-ALL dual-arm SELF-TEST verifier (reads the smoke run's JSONs + log).

Checks, per the pre-registered gate list:
  (a) activation marker  '[CBT-VID] vid='  count > 0, and the allcats=on suffix
      is present (proves --cbt_all_cats reached the operator), and
      '[AB-VID] vid=' count == videos processed (FM17), '[AB-ERR]' absent.
  (b) ab_arm_off_boundaries present in every produced JSON with
      len(ab_arm_off_boundaries) == len(anomalies)  (FM20: written AND grepped).
  (c) ARM INDEPENDENCE: a video where CBT-ALL fired 0 times must be byte-
      identical between arms; a video where it fired must DIFFER.
  (d) FM2 count freeze: len(arm-off) == len(arm-on) on every video, and
      ab_arm_off_confidences index-aligned and equal (CBT is conf-frozen).

Usage: python3.14 analysis_v203_selftest.py <results_dir>
"""
import json
import os
import re
import sys
from pathlib import Path


def main(d):
    d = Path(d)
    log = d / 'run.log'
    txt = log.read_text(errors='replace') if log.exists() else ''
    n_cbtvid = txt.count('[CBT-VID] vid=')
    n_allcats = txt.count('allcats=on')
    n_abvid = txt.count('[AB-VID] vid=')
    n_abfire = txt.count('[AB-FIRE] vid=')
    n_aberr = txt.count('[AB-ERR]')
    n_coll = txt.count('[AB-COLLISION]')
    # per-video CBT fires attributable to the 9 extended cats
    allfired = {}
    for m in re.finditer(r'\[CBT-VID\] vid=(\S+).*?allfired=(\d+)', txt):
        # [CBT-VID] logs the basename WITH extension; the JSON side keys on the
        # stem -> normalise both to the stem.
        k = m.group(1).rsplit('.', 1)[0]
        allfired[k] = allfired.get(k, 0) + int(m.group(2))
    print(f'(a) [CBT-VID] vid= : {n_cbtvid}   allcats=on : {n_allcats}')
    print(f'(a) [AB-VID] vid=  : {n_abvid}   [AB-FIRE]: {n_abfire}   '
          f'[AB-ERR]: {n_aberr}   [AB-COLLISION]: {n_coll}')
    print(f'(a) per-video allfired (extended-cat CBT fires): {allfired}')

    files = sorted(f for f in d.glob('*.json')
                   if f.name != 'evaluation_report.json')
    ok_b = ok_c_zero = ok_c_fire = ok_d = True
    zero_seen = fire_seen = False
    for f in files:
        j = json.load(open(f))
        stem = os.path.basename(str(j.get('video_path', f.stem)))
        stem = stem.rsplit('.', 1)[0]
        an = j.get('anomalies') or []
        off = j.get('ab_arm_off_boundaries')
        offc = j.get('ab_arm_off_confidences')
        if off is None:
            print(f'  FAIL(b) {stem}: ab_arm_off_boundaries MISSING')
            ok_b = False
            continue
        if len(off) != len(an):
            print(f'  FAIL(b) {stem}: len(off)={len(off)} != len(anomalies)={len(an)}')
            ok_b = ok_d = False
        n_diff = sum(1 for a, w in zip(an, off)
                     if abs(float(a['t_start']) - float(w[0])) > 1e-9
                     or abs(float(a['t_end']) - float(w[1])) > 1e-9)
        n_cdiff = 0
        if offc is not None:
            if len(offc) != len(an):
                print(f'  FAIL(d) {stem}: len(off_conf)={len(offc)} != {len(an)}')
                ok_d = False
            n_cdiff = sum(1 for a, c in zip(an, offc)
                          if abs(float(a.get('confidence', 0.0)) - float(c)) > 1e-9)
        fired = allfired.get(stem, 0)
        print(f'  {stem}: ndet={len(an)} armoff_len={len(off)} '
              f'cbt_all_fires={fired} win_diff={n_diff} conf_diff={n_cdiff}')
        if fired == 0:
            zero_seen = True
            if n_diff != 0:
                print(f'  FAIL(c) {stem}: 0 fires but {n_diff} windows differ')
                ok_c_zero = False
        else:
            fire_seen = True
            if n_diff == 0:
                print(f'  FAIL(c) {stem}: {fired} fires but arms identical')
                ok_c_fire = False
        if n_cdiff != 0:
            print(f'  FAIL(d) {stem}: CBT must be confidence-frozen, '
                  f'{n_cdiff} confidences differ')
            ok_d = False

    ok_a = (n_cbtvid > 0 and n_allcats > 0 and n_aberr == 0
            and n_abvid == len(files))
    print(f'\n(a) marker/activation      : {"PASS" if ok_a else "FAIL"}')
    print(f'(b) paired field len-match : {"PASS" if ok_b else "FAIL"}')
    print(f'(c) arm independence       : '
          f'{"PASS" if (ok_c_zero and ok_c_fire and zero_seen and fire_seen) else "FAIL"}'
          f'  (0-fire video seen={zero_seen}, firing video seen={fire_seen})')
    print(f'(d) FM2 count/conf freeze  : {"PASS" if ok_d else "FAIL"}')
    allok = ok_a and ok_b and ok_c_zero and ok_c_fire and zero_seen and fire_seen and ok_d
    print(f'SELFTEST: {"ALL PASS" if allok else "NOT ALL PASS"}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else 'results/batch_eval_v203_smoke')
