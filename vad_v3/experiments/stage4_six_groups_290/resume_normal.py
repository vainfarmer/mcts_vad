"""在有源码/数值核验的情况下，复用已完成视频并以逐窗显存方案继续补测。"""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import time
import zipfile
from run_normal import HERE,ROOT,CODE,PREVIOUS,read,write,sha,install_recording_wrapper


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/'vad_v3/webapp/.env',override=False,encoding='utf-8-sig')
    import experiment
    original_prepare=experiment.prepare_manifest
    old=read(HERE/'normal_results/metadata/manifest.json')
    validation=read(HERE/'streaming_validation.json')
    assert validation['scores_identical'] and validation['max_abs_error']==0
    assert validation['streaming_source_sha256']==sha(CODE/'videomae_classifier.py')
    # 保证除已验证的显存优化外，其他推理文件仍是本次初始版本。
    for name,digest in old['source_sha256'].items():
        if name!='videomae_classifier.py':assert sha(CODE/name)==digest,name
    with zipfile.ZipFile(HERE/'executed_source.zip') as archive:
        before=archive.read('video_anomaly_detection/videomae_classifier.py').decode('utf-8').replace('\r\n','\n')
    expected=before.replace('        tensors = [self._preprocess(frames[s:e]) for s, e in windows]\n','').replace(
        '            for i, tensor in enumerate(tensors):','            for s, e in windows:\n                tensor = self._preprocess(frames[s:e])').replace(
        '                s, e = windows[i]\n','').replace(
        '                scores.append(1.0 - normal_prob)','                scores.append(1.0 - normal_prob)\n                del tensor, logits')
    assert ast.dump(ast.parse(expected))==ast.dump(ast.parse((CODE/'videomae_classifier.py').read_bytes()))
    destination=HERE/'normal_resume_results'
    def prepare_with_reuse(path,config,clients,videos,runtime_settings=None):
        manifest=original_prepare(path,config,clients,videos,runtime_settings)
        for key in ('config','models','api_settings','local_model_sha256','packages','cached_model_revisions','videos','runtime_settings'):
            assert manifest[key]==old[key],key
        origins={}
        for source in sorted((HERE/'normal_results').glob('Normal*.json')):
            result=read(source);assert result['experiment_fingerprint']==old['fingerprint']
            origin=dict(source_result=str(source),source_sha256=sha(source),
                source_fingerprint=old['fingerprint'],kind='completed_before_memory_optimization',
                validation=str(HERE/'streaming_validation.json'))
            result['result_origin']=origin;result['experiment_fingerprint']=manifest['fingerprint']
            target=destination/source.name
            if target.exists():assert read(target)==result,source.name
            else:write(target,result)
            origins[source.stem]=origin
        write(HERE/'resume_provenance.json',dict(imported=len(origins),remaining=150-len(origins),
            origins=origins,streaming_validation=validation,source_manifest_sha256=sha(HERE/'normal_results/metadata/manifest.json'),
            resume_manifest_fingerprint=manifest['fingerprint'],runner_sha256=sha(__file__),
            reason='Long-video eager GPU window inputs saturated VRAM. Interrupted unfinished videos, preserved every completed output; resume with streaming input tensor lifetime.'))
        # 原包装的校验入口固定读取初始 manifest；模型、配置等已逐项再次核验。
        return manifest
    experiment.prepare_manifest=prepare_with_reuse
    install_recording_wrapper(read(PREVIOUS/'results/metadata/manifest.json'))
    command=read(HERE/'COMMAND.json')[2]['batch_argv']
    command[command.index('--output_dir')+1]=str(destination)
    write(HERE/'RESUME_COMMAND.json',[sys.executable,str(Path(__file__)),dict(batch_argv=command)])
    archive_path=HERE/'resume_executed_source.zip'
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as z:
            for p in CODE.glob('*.py'):z.write(p,'video_anomaly_detection/'+p.name)
            for p in (ROOT/'evaluation_tools').glob('*.py'):z.write(p,'evaluation_tools/'+p.name)
            for name in ('run_normal.py','resume_normal.py','compare.py'):z.write(HERE/name,'experiment/'+name)
    helper=HERE.parent/'stage4_acceptance_ablation/run_replay.py'
    spec=importlib.util.spec_from_file_location('resume_gpu_slot',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        with module.gpu_slot():
            sys.argv=command
            runpy.run_path(str(CODE/'batch_evaluate.py'),run_name='__main__')
        write(HERE/'resume_exit_status.json',dict(status='completed',finished_at=time.time()))
    except BaseException as exc:
        write(HERE/'resume_exit_status.json',dict(status='failed',error_type=type(exc).__name__,finished_at=time.time()))
        raise


if __name__=='__main__':main()
