"""Read-only low-memory inventory and formal Object10 protocol evidence.

Stdlib only: read NPY headers and small JSON/trace records, never map the large
feature tensors, import Torch, start GPU jobs, delete data or alter old results.
Stream the JSON to a local file for independent CPU threshold replay.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def npy_header(path):
    with path.open('rb') as f:
        if f.read(6)!=b'\x93NUMPY':raise ValueError('invalid NPY')
        version=f.read(2)
        size=2 if version==b'\x01\x00' else 4 if version in (b'\x02\x00',b'\x03\x00') else 0
        if not size:raise ValueError('unsupported NPY header')
        length=struct.unpack('<H' if size==2 else '<I',f.read(size))[0]
        if length>65536:raise ValueError('oversized NPY header')
        meta=ast.literal_eval(f.read(length).decode('utf8' if version==b'\x03\x00' else 'latin1'))
        offset=f.tell()
    st=path.stat()
    return {'shape':list(meta['shape']),'dtype':meta['descr'],'fortran_order':meta['fortran_order'],
            'header_bytes':offset,'file_bytes':st.st_size,'allocated_bytes':st.st_blocks*512}


def directories(path):
    p=subprocess.run(['du','-x','-B1','--max-depth=1',str(path)],text=True,capture_output=True)
    if p.returncode:raise RuntimeError('du failed: '+p.stderr)
    return sorted([{'bytes':int(l.split('\t')[0]),'path':l.split('\t')[1]}
                   for l in p.stdout.splitlines() if '\t'in l],key=lambda r:-r['bytes'])


def resources():
    group=Path('/sys/fs/cgroup')
    stat={k:int(v) for k,v in (l.split() for l in (group/'memory.stat').read_text().splitlines())}
    limit=(group/'memory.max').read_text().strip()
    processes=subprocess.run(['ps','-eo','pid,ppid,rss,comm','--sort=-rss'],capture_output=True,text=True,check=True).stdout.splitlines()
    return {'memory_limit_bytes':int(limit) if limit!='max' else None,
            'memory_current_bytes':int((group/'memory.current').read_text()),'memory_stat_bytes':stat,
            'top_processes_rss':processes[:15],
            'disks':{p:{'total_bytes':shutil.disk_usage(p).total,'used_bytes':shutil.disk_usage(p).used,
                        'free_bytes':shutil.disk_usage(p).free} for p in ('/','/root/autodl-tmp')},
            'host_meminfo_warning':'/proc/meminfo is host-wide and is not container entitlement'}


def trials(root,model_names):
    plan=read(root/'plan.json');rows=[]
    for f in sorted((root/'trials').glob('*.json')):
        r=read(f);c=r['case']
        if r['status']!='completed' or c['method'] not in model_names:continue
        directory=root/'attempts'/c['id']/('attempt_%03d'%r['attempt'])
        trace=directory/'monitor_steps.jsonl'
        logs=[json.loads(l) for l in trace.read_text().splitlines() if l.strip()] if trace.exists() else []
        scored=[{k:row.get(k) for k in ('step','score','threshold','trigger','interrupted','prediction_crossed_replan')}
                for row in logs if type(row.get('score')) in (int,float)]
        rows.append({'case':c,'metrics':r['result'],'trace':scored,
            'prediction_cancellations_logged':sum(sum(row.get(k,0) or 0 for k in
                ('cancelled_predictions','cancelled_on_interrupt','cancelled_on_new_plan')) for row in logs),
            'source_trial_sha256':sha(f),'trace_sha256':sha(trace) if trace.exists() else None})
    return {'root':str(root),'protocol':plan['protocol'],'plan_sha256':sha(root/'plan.json'),
            'execution':read(root/'execution.json'),'rows':rows}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--base',type=Path,default=Path('/root/autodl-tmp/corrector'));a=p.parse_args()
    base=a.base;code=base/'dev/j36m-35ecdd7488bf'
    config_path=code/'studies/policy_generalization/corrector/j36m/data70.json'
    cfg=read(config_path);specs=cfg['episodes'];arrays=[];connector=[]
    for spec in specs:
        for name in ('block_02','block_08'):
            f=Path(spec['raw_path'])/(name+'.fp16.npy');header=npy_header(f)
            if header['shape']!=[spec['frames'],2,1024,768] or header['dtype']!='<f2':raise ValueError('raw feature contract differs')
            arrays.append({'episode':spec['episode'],'layer':name,'path':str(f),**header})
        folder=Path(cfg['features_root'])/('episode_'+str(spec['episode']))
        for name in ('z_q.npy','z_scale.npy'):
            connector.append({'episode':spec['episode'],'path':str(folder/name),**npy_header(folder/name)})
    manifests=read(base/'object10/manifest.json')
    frames=sum(r['frames'] for r in manifests['records'])
    storage={'data70_config_sha256':sha(config_path),'episodes70':len(specs),'frames70':sum(r['frames'] for r in specs),
             'splits70':dict(Counter(r['split'] for r in specs)),'raw_arrays':arrays,'connector_arrays70':connector,
             'raw70_file_bytes':sum(r['file_bytes'] for r in arrays),'raw70_allocated_bytes':sum(r['allocated_bytes'] for r in arrays),
             'connector70_file_bytes':sum(r['file_bytes'] for r in connector),'full_episodes':len(manifests['records']),
             'full_frames':frames,'two_raw_layers_bytes_per_frame':2*2*1024*768*2,
             'full_raw_payload_bytes':frames*2*2*1024*768*2,'full_connector_quantized_payload_bytes':frames*128*(960+2),
             'raw_by_volume':{root:sum(r['file_bytes'] for r in arrays if r['path'].startswith(root+'/'))
                              for root in ('/root/autodl-tmp','/root/multilayer_export_813_full_001','/root/multilayer_export_file213_rest_001')},
             'directory_sizes':{str(root):directories(root) for root in (base,base/'object10',base/'runtime',base/'plus_resources_20261113_001',Path('/root'))}}
    sources={}
    expanded=base/'dev/expanded-4fe058f5c20c'
    for root,marker in [(expanded,'EXPANDED_MANIFEST.json'),(code,'DEPLOYMENT_MANIFEST.json')]:
        sources[str(root/marker)]={'sha256':sha(root/marker),'content':read(root/marker)}
    for f in (expanded/'scripts/corrector_expanded_eval.py',expanded/'scripts/corrector_j36m_object_eval.py',
              code/'scripts/j36m/online.py',code/'scripts/j36m/model.py',code/'scripts/corrector_online_runtime.py',
              code/'scripts/j36m/loss.py',code/'scripts/j36m/extraction.py'):
        sources[str(f)]={'sha256':sha(f),'text':f.read_text()}
    threshold=base/'runtime/vla-corrector-reference/src/lerobot/scripts/lerobot_eval_modified_detection.py'
    tree=ast.parse(threshold.read_text())
    body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('RobustThresholdState','robust_threshold_step')]
    threshold_text='\n\n'.join(ast.unparse(n) for n in body)
    sources[str(threshold)]={'sha256':sha(threshold),'threshold_ast_source':threshold_text}
    formal=trials(Path('/root/corrector_expanded_object_20261113_001/ordinary'),{'j_lvm','j_ogg'})
    old=trials(base/'object10/night_20261109_001/evaluation/mlp_s42/05d55ba9452142be/paired',{'H50','H20','lvm','ogg'})
    cal=read('/root/corrector_j36m_object_20261113_002/j_calibration.json')
    historical={}
    for folder in (base/'j36m_longrun_20261113_001/extraction',base/'j36m_runs/formal_20261113_002',Path('/root/corrector_j36m_longrun_20261113_001')):
        for name in ('status.json','pipeline_status.json'):
            if (folder/name).is_file():historical[str(folder/name)]=read(folder/name)
    protocols={}
    for key,f in [('mlp',base/'object10/night_20261109_001/runs/mlp_s42/protocol.json'),
                  ('joint',base/'j36m_longrun_20261113_001/training/J_s42/protocol.json')]:
        protocols[key]=read(f)
    output={'audited_utc':datetime.now(timezone.utc).isoformat(),'source_writes':False,'gpu_started':False,
            'resources':resources(),'storage':storage,'sources':sources,'formal_j':formal,'historical_four':old,
            'calibration':cal,'training_protocols':protocols,'historical_status':historical,
            'historical_ram_peak_limit':'Extraction reports/supervisor did not record per-process RSS/VmHWM; current RAM is not historic peak.'}
    print(json.dumps(output,ensure_ascii=False,allow_nan=False,separators=(',',':')))


if __name__=='__main__':main()
