"""只读取已完成Hybrid Pilot的三个视频和录制源码；不安装、不写远程、不调用GPU。"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
REMOTE=r'''
import hashlib,io,json,pathlib,subprocess,sys,zipfile
request=json.load(sys.stdin);allowed=pathlib.Path('/root/autodl-tmp/corrector/object10/hybrid_pilot_20261111_001/paired').resolve()
buf=io.BytesIO();manifest=[]
with zipfile.ZipFile(buf,'w',zipfile.ZIP_STORED) as z:
 for item in request:
  path=pathlib.Path(item['path']).resolve();path.relative_to(allowed)
  if path.suffix!='.mp4' or not path.is_file() or path.stat().st_size>40*2**20:raise RuntimeError('视频路径/体积不符')
  raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
  if sha!=item['sha256']:raise RuntimeError('视频与已归档回合SHA不一致')
  name=item['sha256']+'.mp4';z.writestr('videos/'+name,raw);manifest.append({**item,'file':'videos/'+name,'bytes':len(raw)})
 root=pathlib.Path('/root/autodl-tmp/corrector/runtime/lerobot')
 source=root/'src/lerobot/scripts/lerobot_eval.py'
 raw=source.read_bytes();dirty=subprocess.check_output(['git','-C',str(root),'diff','HEAD','--name-only','--',str(source)],text=True).strip()
 if dirty:raise RuntimeError('录制源码已有修改，不能作为冻结版本证据')
 z.writestr('recording/lerobot_eval.py',raw)
 evidence={'git_revision':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'source_sha256':hashlib.sha256(raw).hexdigest(),'tracked_file_clean':True}
 z.writestr('recording/evidence.json',json.dumps(evidence))
 z.writestr('manifest.json',json.dumps({'videos':manifest,'recording':evidence},ensure_ascii=False,indent=2))
sys.stdout.buffer.write(buf.getvalue())
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ssh-host',required=True)
    p.add_argument('--output',type=Path,default=HERE/'.local/imported_video');args=p.parse_args()
    with zipfile.ZipFile(REPO/'outputs/hybrid_pilot_20261111/results.zip') as z:
        selected=[]
        for name in z.namelist():
            if not name.startswith('paired/trials/') or not name.endswith('.json'):continue
            r=json.loads(z.read(name));c=r['case']
            if c['condition']=='libero_object/task_1/init_0' and c['seed']==1000 and c['method'] in ['A','B','C']:
                if r['status']!='completed':raise RuntimeError('目标回合未完成')
                selected.append({'path':r['result']['video_paths'][0],'sha256':r['video_artifacts'][0]['sha256'],
                    'method':c['method'],'success':r['result']['success'],'env_steps':r['result']['env_steps'],
                    'trial_sha256':hashlib.sha256(z.read(name)).hexdigest()})
    if len(selected)!=3:raise RuntimeError('三个已完成自然回合没有全部找到')
    if args.output.exists():raise RuntimeError('导入目录已存在，拒绝覆盖；无需重复导入')
    args.output.parent.mkdir(exist_ok=True,parents=True);partial=args.output.with_suffix('.zip.partial')
    command=['ssh','-T','-o','ConnectTimeout=15','-o','ServerAliveInterval=30',args.ssh_host,shlex.join(['python3','-c',REMOTE])]
    print('只读取回3个已完成回合视频和录制源码；请在终端完成SSH认证。',flush=True)
    with partial.open('xb') as file:result=subprocess.run(command,input=json.dumps(selected).encode(),stdout=file)
    if result.returncode:raise RuntimeError('SSH读取失败，未导入半成品')
    with zipfile.ZipFile(partial) as z:
        if z.testzip():raise RuntimeError('ZIP损坏')
        manifest=json.loads(z.read('manifest.json'))
        expected={'manifest.json','recording/lerobot_eval.py','recording/evidence.json'}|{v['file'] for v in manifest['videos']}
        if set(z.namelist())!=expected:raise RuntimeError('不接受未知文件')
        for v in manifest['videos']:
            if hashlib.sha256(z.read(v['file'])).hexdigest()!=v['sha256']:raise RuntimeError('视频SHA不符')
        args.output.mkdir()
        for name in expected:
            path=args.output/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(z.read(name))
    archive=args.output.parent/'imported_video_source.zip';partial.replace(archive)
    print(json.dumps({'imported':3,'output':str(args.output),'bytes':archive.stat().st_size,'gpu_work':False},ensure_ascii=False))


if __name__=='__main__':main()
