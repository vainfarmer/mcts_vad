from pathlib import Path
import os, sys, subprocess, json

here = Path(__file__).resolve().parent;
code = here.parents[1] / 'video_anomaly_detection'
for line in (code / '.env').read_text(encoding='utf-8-sig').splitlines():
    if line.strip() and not line.lstrip().startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        if k.strip() == 'DASHSCOPE_API_KEY': os.environ[k.strip()] = v.strip().strip('\"\x27')
assert os.environ.get('DASHSCOPE_API_KEY'), 'Missing API key'
os.environ['PYTHONIOENCODING'] = 'utf-8';
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(code))
from batch_evaluate import select_videos

selected = select_videos('F:/dataset/ucf_crime', 290)
assert len(selected) == 290 and selected.video_path.nunique() == 290, 'Expected full 290-video test set'
assert all((Path('F:/dataset/ucf_crime') / path).is_file() for path in selected.video_path), 'Missing video'
(here / 'selected_videos.json').write_text(selected.to_json(orient='records', indent=2), encoding='utf-8')
args = [sys.executable, '-u', str(code / 'batch_evaluate.py'), '--dataset_dir', 'F:/dataset/ucf_crime', '--output_dir',
        str(here / 'results'), '--detect_model', 'qwen3-vl-plus', '--classify_model', 'qwen3-vl-plus',
        '--max_per_category', '290', '--concurrency', '2', '--clip_model', 'ViT-L-14', '--device', 'cuda', '--mcts_mode',
        'refine', '--mcts_paired_eval', '--mcts_trace', '--mcts_action_set', 'shrink_expand_shift',
        '--mcts_citation_repair_attempts', '1', '--mcts_trigger_policy', 'all_valid', '--mcts_acceptance_policy',
        'relative_nonloss', '--mcts_max_images_per_evaluation', '32', '--mcts_max_boundary_gap_seconds', '1.5',
        '--mcts_max_evidence_gap_seconds', '3.0', '--mcts_max_candidates_per_video', '4',
        '--mcts_max_vlm_evaluations_per_candidate', '10', '--mcts_max_vlm_evaluations_per_video', '40']
args += ['--mcts_final_nms_enabled', '--api_concurrency', '4']
(here / 'COMMAND.json').write_text(json.dumps(args, indent=2), encoding='utf-8')
with (here / 'v217_console.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(args, cwd=code, stdout=log, stderr=subprocess.STDOUT)
(here / 'exit_status.json').write_text(json.dumps({'returncode': result.returncode}), encoding='utf-8')
sys.exit(result.returncode)
