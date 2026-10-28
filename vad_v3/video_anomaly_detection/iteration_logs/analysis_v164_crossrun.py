import sys
sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth, load_predictions, compute_temporal_map

LABEL = '/Users/zzl/Movies/ucf-crime/data'
HARD = {'Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion'}
V160 = '/Users/zzl/PycharmProjects/vad/results/batch_eval_v160_full'
BUE = '/Users/zzl/PycharmProjects/vad/results/batch_eval_full290_bue'

gt = load_ground_truth(LABEL, 'test')
p160 = load_predictions(V160)
pbue = load_predictions(BUE)


def sub(d, cats, invert=False):
    return {k: v for k, v in d.items()
            if (v['category'] in cats) != invert}


def m(gt_sub, pred):
    ps = {k: pred[k] for k in gt_sub if k in pred}
    return compute_temporal_map(gt_sub, ps)['mAP']


gt_hard = sub(gt, HARD)
gt_nonhard = sub(gt, HARD, invert=True)

full_160 = compute_temporal_map(gt, {k: p160[k] for k in gt if k in p160})['mAP']
full_bue = compute_temporal_map(gt, {k: pbue[k] for k in gt if k in pbue})['mAP']

h160 = m(gt_hard, p160)
hbue = m(gt_hard, pbue)
n160 = m(gt_nonhard, p160)
nbue = m(gt_nonhard, pbue)

print('FULL290  v160=%.5f  bue=%.5f  delta=%+.5f' % (full_160, full_bue, full_160 - full_bue))
print('HARD(88) v160=%.5f  bue=%.5f  delta=%+.5f' % (h160, hbue, h160 - hbue))
print('NONHARD  v160=%.5f  bue=%.5f  delta=%+.5f' % (n160, nbue, n160 - nbue))
print('crossrun_hard_delta=%.5f' % (h160 - hbue))
print('crossrun_nonhard_delta=%.5f' % (n160 - nbue))

# per-category AP (single-cat global mAP as a proxy per cat)
print('--- per hard cat ---')
for c in sorted(HARD):
    g = sub(gt, {c})
    print('%-14s v160=%.5f bue=%.5f d=%+.5f (n=%d)' % (
        c, m(g, p160), m(g, pbue), m(g, p160) - m(g, pbue), len(g)))
