"""只读取一组已完成Horizon旧回合的6段视频；复用原导入校验，不运行模型。"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import zipfile

from data import Catalog,HERE
from import_review import REMOTE,validate

METHODS=['H50','H20','H10','H5','H2','H1']


def selection():
    catalog=Catalog();groups=defaultdict(dict)
    try:
        for r in catalog.runs.values():
            if r['experiment']!='smolvla_horizon_pilot' or r['forced'] or r['status']!='completed':continue
            groups[r['condition'],r['seed']][r['method']]=r
        candidates=[g for g in groups.values() if all(m in g and g[m]['_record'].get('video_artifacts') for m in METHODS)]
        if not candidates:raise RuntimeError('没有完整六方法旧视频记录，不启动补测或猜路径')
        group=min(candidates,key=lambda g:(g['H50']['success'],g['H20']['success'],not g['H1']['success'],g['H50']['condition'],g['H50']['seed']))
        result=[]
        for method in METHODS:
            r=group[method];raw=r['_source'].read(r['member']);artifact=r['_record']['video_artifacts'][0]
            trial_sha=hashlib.sha256(raw).hexdigest()
            if trial_sha in catalog.video_trial_hash.get(artifact['sha256'],set()):continue
            result.append({'member':r['member'],'trial_path':'/root/autodl-tmp/corrector/object10/horizon_pilot_20261111_001/'+r['member'],
                'trial_sha256':trial_sha,'sha256':artifact['sha256'],'model':'SmolVLA','method':method,
                'condition':r['condition'],'seed':r['seed'],'success':r['success'],'env_steps':r['steps']})
        return result
    finally:catalog.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ssh-host',required=True);p.add_argument('--control-path',type=Path);p.add_argument('--plan',action='store_true');a=p.parse_args()
    requested=selection()
    if a.plan or not requested:print(json.dumps({'requested':requested,'already_complete':not requested},ensure_ascii=False,indent=2));return
    state=HERE/'.local'
    state.mkdir(parents=True,exist_ok=True);(state/'import_batches').mkdir(exist_ok=True)
    if shutil.disk_usage(state).free<256*2**20:raise RuntimeError('本地不足256MiB，不取回，不清理旧文件')
    destination=state/'import_batches'/('horizon_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));partial=destination.with_suffix('.zip.partial')
    # 仅扩展原只读导入器的明确目录白名单；录制源码、原trial及视频SHA校验完全复用。
    remote=REMOTE.replace("roots={'night_20261109_001','action_followup_20261109_001'}","roots={'horizon_pilot_20261111_001'}")
    ssh=['ssh','-T','-o','ConnectTimeout=15']
    if a.control_path:ssh+=['-S',str(a.control_path),'-o','BatchMode=yes']
    ssh+=[a.ssh_host,shlex.join(['python3','-c',remote])]
    with partial.open('xb') as f:r=subprocess.run(ssh,input=json.dumps(requested).encode(),stdout=f)
    if r.returncode:raise RuntimeError('只读取回失败，保留现场；未启动GPU')
    manifest=validate(partial,requested)
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(partial) as z:
        for name in z.namelist():
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    archive=partial.with_suffix('');partial.replace(archive)
    print(json.dumps({'imported_bindings':len(manifest['videos']),'skipped':manifest.get('skipped',[]),'bytes':archive.stat().st_size,
        'output':str(destination),'gpu_work':False,'remote_read_only':True,'restart_local_replay_to_index':True},ensure_ascii=False))


if __name__=='__main__':main()
