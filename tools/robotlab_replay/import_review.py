"""追加导入六段基线视频及截图中A10回合的独立录制绑定；远程只读。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REVISION = '7e241bd630a3719a56157a497ce5d08f244784f1'
SOURCE_SHA = 'fddec10c709401dd863814091d525fcd54d41cc6ccf51f7ef8300459cf69b873'

# 不执行任何实验代码，不在服务器创建文件。请求限定在已有冻结实验目录。
REMOTE = r'''
import hashlib,io,json,pathlib,subprocess,sys,zipfile
request=json.load(sys.stdin)
base=pathlib.Path('/root/autodl-tmp/corrector/object10').resolve()
roots={'night_20261109_001','action_followup_20261109_001'}
if not 1<=len(request)<=10:raise RuntimeError('请求回合数量异常')
recorder=pathlib.Path('/root/autodl-tmp/corrector/runtime/lerobot')
source=recorder/'src/lerobot/scripts/lerobot_eval.py';source_raw=source.read_bytes()
revision=subprocess.check_output(['git','-C',str(recorder),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(recorder),'diff','HEAD','--name-only','--',str(source)],text=True).strip()
source_sha=hashlib.sha256(source_raw).hexdigest()
if dirty or revision!='7e241bd630a3719a56157a497ce5d08f244784f1' or source_sha!='fddec10c709401dd863814091d525fcd54d41cc6ccf51f7ef8300459cf69b873':
 raise RuntimeError('录制版本/SHA不符，停止，不用当前源码替代旧录制协议')
videos=[];skipped=[];written=set();total=0;buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_STORED) as z:
 for item in request:
  trial=pathlib.Path(item['trial_path']).resolve();relative=trial.relative_to(base)
  if relative.parts[0] not in roots or trial.parent.name!='trials' or trial.suffix!='.json':raise RuntimeError('回合路径越界')
  raw=trial.read_bytes();record=json.loads(raw)
  if hashlib.sha256(raw).hexdigest()!=item['trial_sha256'] or record.get('status')!='completed':raise RuntimeError('原trial已改变或未完成')
  evaluation=trial.parent.parent;plan=json.loads((evaluation/'plan.json').read_text())
  if plan['protocol'].get('lerobot_revision')!=revision:raise RuntimeError('回合的录制版本不一致')
  artifact=next((v for v in record.get('video_artifacts',[]) if v.get('sha256')==item['sha256']),None)
  if not artifact:raise RuntimeError('原回合未登记该视频')
  video=(evaluation/artifact['path']).resolve();video.relative_to(evaluation)
  if video.suffix!='.mp4':raise RuntimeError('不接受非视频文件')
  if not video.exists():skipped.append({'member':item['member'],'reason':'原端已无该视频'});continue
  if video.stat().st_size>40*2**20:raise RuntimeError('单视频超过40MiB')
  payload=video.read_bytes();sha=hashlib.sha256(payload).hexdigest()
  if sha!=item['sha256']:raise RuntimeError('视频SHA改变')
  name='videos/'+sha+'.mp4'
  if name not in written:
   total+=len(payload)
   if total>80*2**20:raise RuntimeError('导入总量超过80MiB')
   z.writestr(name,payload);written.add(name)
  videos.append({**item,'file':name,'bytes':len(payload),'remote_trial_verified':True,
    'recording_revision':revision,'recording_source_sha256':source_sha})
 evidence={'git_revision':revision,'source_sha256':source_sha,'tracked_file_clean':True}
 z.writestr('recording/lerobot_eval.py',source_raw)
 z.writestr('recording/evidence.json',json.dumps(evidence))
 z.writestr('manifest.json',json.dumps({'videos':videos,'skipped':skipped,'recording':evidence,'remote_read_only':True},ensure_ascii=False,indent=2))
sys.stdout.buffer.write(buf.getvalue())
'''


def selection():
    """固定两组完整失败对照，避免事后将缺视频的成功回合冒充可回放。"""
    archive = REPO/'outputs/corrector_audit_20261111/results_bundle.zip'
    selected = []
    with zipfile.ZipFile(archive) as z:
        roots = json.loads(z.read('bundle_manifest.json'))['roots']
        for member in sorted(z.namelist()):
            if '/paired/trials/' not in member or not member.endswith('.json'):
                continue
            raw = z.read(member); record = json.loads(raw); case = record['case']
            baseline = (member.startswith('night/evaluation/mlp_s42/') and
                        case['condition'] in {'libero_object/task_2/init_2','libero_object/task_9/init_2'} and
                        case['method'] in {'H50','H20','ogg'} and case['seed'] == 1000)
            alias = (member.startswith('actions/experiments/evaluation/transformer-a10-h1_s42/') and
                     case['condition'] == 'libero_object/task_1/init_0' and case['method'] == 'ogg' and case['seed'] == 1000)
            if not (baseline or alias): continue
            if record['status'] != 'completed' or not record.get('video_artifacts'):
                raise ValueError('预选回合尚未完成或没有保留视频：'+member)
            video = record['video_artifacts'][0]
            root, relative = member.split('/',1)
            selected.append({'member':member,'trial_path':roots[root]+'/'+relative,
                'trial_sha256':hashlib.sha256(raw).hexdigest(),'sha256':video['sha256'],
                'model':'MLP' if baseline else 'A10-H1','method':case['method'],
                'condition':case['condition'],'seed':case['seed'],'success':record['result']['success'],
                'env_steps':record['result']['env_steps']})
    if len(selected) != 7:
        raise ValueError(f'预选回合应为7条，实际{len(selected)}；停止，不猜测路径')
    return selected


def validate(path, requested):
    """在写入新批次前检查路径、预算、来源绑定及所有内容哈希。"""
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(set(z.namelist())) or sum(i.file_size for i in z.infolist()) > 82*2**20:
            raise ValueError('ZIP重复成员或超出预算')
        if z.testzip():raise ValueError('ZIP校验失败')
        manifest = json.loads(z.read('manifest.json'))
        expected = {'manifest.json','recording/lerobot_eval.py','recording/evidence.json'}
        known = {(r['trial_sha256'],r['sha256']):r for r in requested}
        for v in manifest['videos']:
            original = known.get((v['trial_sha256'],v['sha256']))
            if original is None or any(v.get(k) != val for k,val in original.items()):
                raise ValueError('回包包含未请求或被改写的回合')
            if v['file'] != 'videos/'+v['sha256']+'.mp4' or not v.get('remote_trial_verified'):
                raise ValueError('视频路径或原回合绑定不符')
            expected.add(v['file'])
            if hashlib.sha256(z.read(v['file'])).hexdigest() != v['sha256']:raise ValueError('视频SHA不符')
        if set(z.namelist()) != expected:raise ValueError('ZIP含未知文件')
        evidence = json.loads(z.read('recording/evidence.json'))
        if (evidence.get('git_revision') != REVISION or not evidence.get('tracked_file_clean') or
            evidence.get('source_sha256') != SOURCE_SHA or hashlib.sha256(z.read('recording/lerobot_eval.py')).hexdigest() != SOURCE_SHA):
            raise ValueError('录制证据不符')
        return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh-host',required=True,help='本机SSH配置别名，仅用于本次只读连接')
    p.add_argument('--plan',action='store_true',help='只在本地预览选择，不连接SSH')
    args=p.parse_args();requested=selection()
    if args.plan:
        print(json.dumps([{k:v for k,v in r.items() if k not in {'trial_path','member'}} for r in requested],ensure_ascii=False,indent=2));return
    state=HERE/'.local';destination=state/'import_batches'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    partial=destination.with_suffix('.zip.partial');partial.parent.mkdir(parents=True,exist_ok=True)
    command=['ssh','-T','-o','ConnectTimeout=15','-o','ServerAliveInterval=30',args.ssh_host,shlex.join(['python3','-c',REMOTE])]
    print('仅取回7条旧回合的视频/录制证据；不写远程、不启动GPU。请在终端完成SSH认证。',flush=True)
    with partial.open('xb') as file:result=subprocess.run(command,input=json.dumps(requested).encode(),stdout=file)
    if result.returncode:raise RuntimeError('SSH读取失败，未导入半成品；旧资料保持不变')
    manifest=validate(partial,requested)
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(partial) as z:
        for name in z.namelist():
            # validate已限定完整成员集合，再显式拒绝越界。
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:raise ValueError('路径越界')
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    archive=partial.with_suffix('');partial.replace(archive)
    print(json.dumps({'imported_bindings':len(manifest['videos']),'unique_videos':len({v['sha256'] for v in manifest['videos']}),
        'skipped':manifest.get('skipped',[]),'output':str(destination),'bytes':archive.stat().st_size,
        'restart_local_replay_to_index':True,'gpu_work':False},ensure_ascii=False))


if __name__=='__main__':main()
