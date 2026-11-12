"""验证完整样本库并追加到Replay私有导入批次；只用标准库，不加载pickle/Torch。"""
import ast
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import shutil
import stat
import struct
import zipfile

REVISION='7e241bd630a3719a56157a497ce5d08f244784f1'
RECORDER_SHA='fddec10c709401dd863814091d525fcd54d41cc6ccf51f7ef8300459cf69b873'


def validate_numeric_npz(raw):
    sizes={'<f4':4,'<f8':8,'<i4':4,'<i8':8,'|u1':1}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if not 1<=len(z.infolist())<=20 or sum(i.file_size for i in z.infolist())>64*2**20:raise ValueError('NPZ预算异常')
        for item in z.infolist():
            if '/' in item.filename or not item.filename.endswith('.npy'):raise ValueError('NPZ成员不是白名单数值数组')
            with z.open(item) as f:
                if f.read(6)!=b'\x93NUMPY':raise ValueError('非法NPY头')
                version=f.read(2)
                if version not in [b'\x01\x00',b'\x02\x00']:raise ValueError('未知NPY版本')
                count=2 if version[0]==1 else 4;length=struct.unpack('<H' if count==2 else '<I',f.read(count))[0]
                if length>65536:raise ValueError('NPY头过大')
                header=ast.literal_eval(f.read(length).decode('latin1'));shape=header['shape'];dtype=header['descr']
                if dtype not in sizes or header['fortran_order'] is not False or not isinstance(shape,tuple) or not 1<=len(shape)<=5 or any(type(d) is not int or d<1 for d in shape):raise ValueError('NPY不是安全的普通数值数组')
                n=1
                for d in shape:n*=d
                if n*sizes[dtype]!=item.file_size-(6+2+count+length):raise ValueError('NPY尺寸与实际字节数不符')


def install_bundle(archive,state):
    archive=Path(archive);state=Path(state);h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    sha=h.hexdigest()
    target=state/'import_batches'/('casebank_'+sha[:16])
    if target.exists():
        if not (target/'manifest.json').is_file():raise ValueError('发现未完成导入批次，保留现场，不宣称已导入')
        return {'already_imported':True,'output':str(target),'source_sha256':sha}
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=set(z.namelist())
        if len(names)!=len(infos) or len(names)>100000 or sum(i.file_size for i in infos)>6*2**30:raise ValueError('重复成员或结果包超预算')
        for i in infos:
            p=PurePosixPath(i.filename)
            if '\\' in i.filename or ':' in i.filename or p.is_absolute() or '..' in p.parts or not i.filename.startswith('results/') or stat.S_ISLNK(i.external_attr>>16):raise ValueError('非法成员路径/符号链接')
            if i.file_size>40*2**20 or p.suffix not in {'.json','.jsonl','.csv','.md','.mp4','.npz','.py'}:raise ValueError('未知成员类型或单文件超预算')
        if z.testzip():raise ValueError('ZIP校验失败')
        evidence=json.loads(z.read('results/recording/evidence.json'))
        if evidence.get('git_revision')!=REVISION or evidence.get('source_sha256')!=RECORDER_SHA or not evidence.get('tracked_file_clean') or hashlib.sha256(z.read('results/recording/lerobot_eval.py')).hexdigest()!=RECORDER_SHA:raise ValueError('录制证据不匹配')
        videos=[];complete=0
        for member in sorted(n for n in names if n.startswith('results/trials/') and n.endswith('.json')):
            raw=z.read(member);r=json.loads(raw)
            if r.get('status')!='completed':continue
            c=r['case'];n=r['result']['env_steps'];base=f"results/attempts/{c['id']}/attempt_{r['attempt']:03d}/"
            if r['result'].get('capture_schema')!='replay_casebank_v1' or type(n) is not int or not 0<n<=280:raise ValueError('完成回合不是预期采集协议')
            commands=json.loads(z.read(base+'controller_commands.json'));queue=json.loads(z.read(base+'queue_trace.json'))
            if len(commands)!=n or len(queue)!=n or not all(a['step']==i and a['returned'] for i,a in enumerate(commands)):raise ValueError('真实控制记录缺失/步序异常')
            mapping=[json.loads(l) for l in z.read(base+'frame_map.jsonl').decode().splitlines()]
            if len(mapping)!=n or not all(f['env_step']==i and f['frame_index']==i and f['phase']=='before_action' for i,f in enumerate(mapping)):raise ValueError('视频帧映射不完整')
            artifacts=r.get('video_artifacts',[])
            if len(artifacts)!=1:raise ValueError('成功和失败回合都必须有唯一视频')
            for video in artifacts:
                p=PurePosixPath(video['path'])
                if p.is_absolute() or '..' in p.parts or not str(p).startswith(base.removeprefix('results/')) or p.suffix!='.mp4':raise ValueError('视频不是本回合产生')
                path='results/'+str(p)
                if hashlib.sha256(z.read(path)).hexdigest()!=video['sha256']:raise ValueError('视频SHA不符')
                videos.append({'file':path,'sha256':video['sha256'],'trial_sha256':hashlib.sha256(raw).hexdigest(),'path':str(p),
                    'method':c['method'],'condition':c['condition'],'seed':c['seed'],'env_steps':n,'remote_trial_verified':True})
            if c['config']['family']!='baseline' and base+'monitor_steps.jsonl' not in names:raise ValueError('Corrector回合缺逐步trace')
            complete+=1
        execution=json.loads(z.read('results/execution.json'))
        if complete!=execution['completed']:raise ValueError('完成数与原始trial不一致')
        for name in names:
            if name.endswith('.npz'):validate_numeric_npz(z.read(name))
        state.mkdir(parents=True,exist_ok=True)
        indexed=shutil.disk_usage(state).free<sum(i.file_size for i in infos)+2**30
        if indexed:
            # 小盘不复制已经校验的数值数组；仅解出视频，元数据由Source按需读ZIP。
            relative_archive=archive.resolve().relative_to(state.resolve()).as_posix()
            materialized={v['file'] for v in videos}|{'results/recording/evidence.json','results/recording/lerobot_eval.py'}
            if shutil.disk_usage(state).free<sum(i.file_size for i in infos if i.filename in materialized)+512*2**20:
                raise ValueError('连视频与512MiB保底空间也不足；保留ZIP，不解压，不清理旧资料')
        else:materialized=names
        target.mkdir(parents=True,exist_ok=False)
        for info in infos:
            if info.filename not in materialized:continue
            path=target/info.filename;path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,path.open('xb') as dest:shutil.copyfileobj(source,dest)
        recording=target/'recording';recording.mkdir()
        for name in ['evidence.json','lerobot_eval.py']:shutil.copyfile(target/'results/recording'/name,recording/name)
        if indexed:(target/'archive.json').write_text(json.dumps({'archive':relative_archive,'sha256':sha,'storage':'verified_zip_read_only'})+'\n')
        (target/'manifest.json').write_text(json.dumps({'videos':videos,'recording':evidence,'source_sha256':sha},ensure_ascii=False,indent=2)+'\n')
    return {'imported_trials':complete,'videos':len(videos),'output':str(target),'execution_status':execution['status'],
            'source_sha256':sha,'storage':'verified_zip_with_video_cache' if indexed else 'extracted',
            'restart_local_replay_to_index':True}
