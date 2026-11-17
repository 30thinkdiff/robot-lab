"""Audit existing chunk -> selected action -> submitted command, without physics."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from motion_capture import ActionDecoder


def analyze(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,entry in manifest['members'].items():
        p=(root/name).resolve();p.relative_to(root.resolve())
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('historical source SHA changed: '+name)
    decoder=ActionDecoder.from_snapshot(root/'processor');rows=[]
    for case in manifest['audit']:
        cid=case['case_id'];r=json.loads((root/'trials'/(cid+'.json')).read_text())
        d=root/'attempts'/cid/('attempt_%03d'%r['attempt'])
        queue=json.loads((d/'queue_trace.json').read_text());submitted=json.loads((d/'controller_commands.json').read_text())
        calls=json.loads((d/'chunk_calls.json').read_text());chunks={};used={}
        for call in calls:
            p=d/'chunks'/('chunk_%04d.npz'%call['call'])
            if p.stat().st_size>64*1024:raise ValueError('chunk NPZ above 64KiB')
            with zipfile.ZipFile(p) as checked:
                if checked.namelist()!=['normalized_actions.npy'] or sum(i.file_size for i in checked.infolist())>4096:
                    raise ValueError('action chunk member/expanded size differs')
            with np.load(p,allow_pickle=False) as z:
                if z.files!=['normalized_actions']:raise ValueError('unexpected chunk arrays')
                a=z['normalized_actions']
                if a.shape!=(1,50,7) or a.dtype!=np.float32 or not np.isfinite(a).all():
                    raise ValueError('unexpected action chunk shape/dtype')
                chunks[call['call']]=a[0].copy();used[call['call']]=0
        if len(queue)!=len(submitted) or len(queue)!=r['result']['env_steps']:
            raise ValueError('step counts differ')
        max_error=0.;max_selected=0.
        for step,(q,u) in enumerate(zip(queue,submitted)):
            if q['step']!=step or u['step']!=step or not u['returned']:
                raise ValueError('control step order or successful return missing')
            plan=q['plan_id'];slot=used[plan]
            if slot>=50:raise ValueError('consumed beyond generated chunk')
            a=chunks[plan][slot];selected=np.asarray(q['selected_normalized_action']);used[plan]+=1
            delta=float(np.max(np.abs(a-selected)));max_selected=max(max_selected,delta)
            if delta!=0:raise ValueError('recorded selected action differs from chunk/slot')
            decoded=decoder.decode(a);actual=np.asarray(u['raw_controller_action'])
            error=float(np.max(np.abs(decoded-actual)));max_error=max(max_error,error)
            if not np.allclose(decoded,actual,rtol=2e-5,atol=2e-6):
                raise ValueError('frozen output transform differs from submitted command')
        rows.append({'method':case['method'],'condition':case['condition'],'seed':case['seed'],'case_id':cid,
            'success':r['result']['success'],'steps':len(queue),'chunks':len(chunks),'generated_slots':50*len(chunks),
            'consumed_slots':sum(used.values()),'unused_slots':50*len(chunks)-sum(used.values()),
            'selected_normalized_max_abs_error':max_selected,'decoded_submitted_max_abs_error':max_error,
            'per_chunk_consumed':used,'actual_pose_series_available':False,'osc_goal_series_available':False,
            'joint_state_series_available':False,'torque_series_available':False,
            'trajectory_gap':None,'gap_limit':'no per-step actual state/OSC goals; no gap invented from command arrays'})
    return {'schema':'historical_motion_audit_v1','status':'passed','cases':rows,'decoder':decoder.provenance,
        'total_chunks':sum(r['chunks'] for r in rows),'total_submitted_steps':sum(r['steps'] for r in rows),
        'limit':'command-chain consistency only; not task correctness or controller tracking accuracy',
        'gpu_work':False,'physics_replayed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    report=analyze(a.source)
    if a.output.resolve().is_relative_to(a.source.resolve()):raise ValueError('keep derived report outside read-only source')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'report':str(a.output),'cases':len(report['cases']),'chunks':report['total_chunks'],
        'steps':report['total_submitted_steps'],'status':report['status']},ensure_ascii=False))
