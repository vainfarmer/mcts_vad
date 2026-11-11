"""Explicitly synthetic, isolated UI fixture; never imports results into production."""
import json
from pathlib import Path
import shutil
from vad_v3.webapp.settings import atomic_json,effective_config,code_hash,file_hash
from vad_v3.webapp.store import Store
from vad_v3.webapp.tests.test_app import RESULT


def prepare(root,source_video):
    store=Store(root)
    if store.jobs():return
    target=store.path('videos/fixture/input.mp4');target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source_video,target)
    store.add_video('fixture','UI_SYNTHETIC_TEST.mp4','videos/fixture/input.mp4',file_hash(target),
                    dict(duration=10.,fps=30.,frame_count=300))
    job,_=store.create_job('fixture',effective_config(),code_hash(),None)
    _,attempt=store.claim('synthetic-ui-fixture')
    result=json.loads(json.dumps(RESULT))
    result['mcts_trace']=dict(mode='refine',candidates={'0':dict(
        reason='SYNTHETIC_UI_FIXTURE',baseline=[2.,4.],proposed=[2.2,3.8],logical_requests=3,
        nodes=[
            dict(node_id=0,parent_id=None,depth=0,interval=[2.,4.],visits=3,mean_value=.5,evaluation={'reward':.5}),
            dict(node_id=1,parent_id=0,depth=1,interval=[2.2,4.],visits=2,mean_value=.6,evaluation={'reward':.6}),
            dict(node_id=2,parent_id=1,depth=2,interval=[2.2,3.8],visits=1,mean_value=.7,evaluation={'reward':.7})
        ],evaluations=[])})
    folder=store.path(attempt['artifact_dir'])
    atomic_json(folder/'result.json',result)
    atomic_json(folder/'research.json',dict(schema_version=1,exit_reason='SYNTHETIC_UI_FIXTURE',
        classification=dict(category='Fighting',note='Synthetic UI fixture, not model evidence'),evidence=[],
        stages={name:[dict(result['anomalies'][0],t_start=a,t_end=b)] for name,a,b in
                [('discovery_candidates',1.,5.),('post_clip',2.,4.),('post_mcts',2.2,3.8),('final',2.2,3.8)]},
        observations=dict(qscore=dict(base=.5,components=[dict(t_start=2.,t_end=4.,sharpness=.4,
            contrast=.7,center_fit=.8,normalized_quality=1.,before=.8,after=.8)]))))
    store.finish(attempt['id'],attempt['token'],'completed')
