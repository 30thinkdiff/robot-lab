"""Convert completed expanded-J v1 records into an explicitly derived Replay batch.

The input ZIP remains unchanged. No checkpoint/pickle is read. Controller return
flags are inferred from a completed sequential rollout, explicitly not measured
per-step flags. Raw commands and queue lengths were recorded at execution time.
"""
import argparse,copy,hashlib,io,json,sys,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
from corrector_casebank_import import install_bundle,RECORDER_SHA


def convert(source,destination):
    import av
    source=Path(source);destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    provenance={'original_bundle_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'controller_return_flags':'inferred from fully completed sequential eval, not separately measured',
                'queue_keys':'renamed from recorded before/after/chunk_calls; no replayed inference',
                'frame_map':'actual decoded PTS plus verified recording source phase; not FPS-only'}
    with zipfile.ZipFile(source) as z,zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as out:
        def read(n):return json.loads(z.read(n))
        def write(n,v):out.writestr('results/'+n,json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
        recorder=z.read('recording/lerobot_eval.py')
        if hashlib.sha256(recorder).hexdigest()!=RECORDER_SHA:raise ValueError('unknown recorder')
        out.writestr('results/recording/lerobot_eval.py',recorder)
        out.writestr('results/recording/evidence.json',z.read('recording/evidence.json'))
        plan=read('ordinary/plan.json');plan['protocol'].update(study_id='replay_casebank_expanded_ordinary_20261113_derived',
            experiment_label='Object10 · J正式补测（派生索引）',control_fps=20,score_start=24,derivation=provenance)
        records=[]
        for member in sorted(n for n in z.namelist() if n.startswith('ordinary/trials/') and n.endswith('.json')):
            raw=z.read(member);r=json.loads(raw)
            if r['status']!='completed':raise ValueError('only completed v1 phases supported')
            r=copy.deepcopy(r);c=r['case'];c['config'].update(model_label='J-36M',family='joint',action_steps=50)
            r['result']['capture_schema']='replay_casebank_v1'
            r['derivation']={**provenance,'original_trial_sha256':hashlib.sha256(raw).hexdigest(),'original_member':member}
            base=f"attempts/{c['id']}/attempt_{r['attempt']:03d}/";old='ordinary/'+base;n=r['result']['env_steps']
            actions=read(old+'actions.json')['submitted_env_actions'];queue=read(old+'queue_trace.json')
            if len(actions)!=n or len(queue)!=n:raise ValueError('recorded action/queue counts differ')
            previous=0;renamed=[]
            for i,q in enumerate(queue):
                if q['step']!=i or q['chunk_calls']<previous:raise ValueError('queue index/call order invalid')
                renamed.append(dict(q,queue_before=q['before'],queue_after=q['after'],chunk_calls_before=previous,
                                    chunk_calls_after=q['chunk_calls'],plan_id=q['chunk_calls']))
                previous=q['chunk_calls']
            if previous!=r['result']['chunk_calls']:raise ValueError('recorded inference call count differs')
            write(base+'queue_trace.json',renamed);write(base+'executed_actions.json',actions)
            write(base+'controller_commands.json',[{'step':i,'raw_controller_action':a,'returned':True,
                  'returned_evidence':'derived: sequential eval completed, no individual return timestamp'} for i,a in enumerate(actions)])
            actual=read(old+'actual_protocol.json');actual.update(derivation=provenance,monitor_first_due_step=24,
                recording_source_sha256=RECORDER_SHA,recording_phase='frame n before action n; terminal Z_N excluded')
            write(base+'actual_protocol.json',actual)
            video=r['video_artifacts'][0];raw_video=z.read('ordinary/'+video['path'])
            if hashlib.sha256(raw_video).hexdigest()!=video['sha256']:raise ValueError('video SHA mismatch')
            with av.open(io.BytesIO(raw_video)) as container:pts=[float(f.pts*f.time_base) for f in container.decode(video=0)]
            if len(pts)!=n or abs(pts[0])>1e-8 or any(a>=b for a,b in zip(pts,pts[1:])):raise ValueError('video PTS/order/count invalid')
            out.writestr('results/'+video['path'],raw_video)
            out.writestr('results/'+base+'frame_map.jsonl',''.join(json.dumps({'frame_index':i,'env_step':i,'pts_seconds':t,'phase':'before_action'})+'\n' for i,t in enumerate(pts)))
            trace=[]
            for line in z.read(old+'monitor_steps.jsonl').decode().splitlines():
                row=json.loads(line)
                if row.get('event')=='joint_ogg' and 'loss'in row:row['ogg_loss']=row['loss']
                trace.append(row)
            out.writestr('results/'+base+'monitor_steps.jsonl',''.join(json.dumps(row)+'\n' for row in trace))
            write('trials/'+c['id']+'.json',r);out.writestr('results/original_trials/'+c['id']+'.json',raw);records.append(r)
        plan['cases']=[r['case'] for r in records];write('plan.json',plan)
        execution=read('ordinary/execution.json')
        if execution['completed']!=len(records) or execution['status']!='completed':raise ValueError('phase not complete')
        write('execution.json',execution);write('derivation.json',provenance)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--state',type=Path,default=HERE/'.local');a=p.parse_args()
    path=convert(a.source,a.destination);print(json.dumps(install_bundle(path,a.state),ensure_ascii=False))
