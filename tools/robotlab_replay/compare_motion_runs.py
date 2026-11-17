"""Same-case capture off/on acceptance; never infer motion from video pixels."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())


def frame_hash(path):
    import av
    h=hashlib.sha256();count=0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):h.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
    return {'rgb_sha256':h.hexdigest(),'frames':count}


def compare(reference,captured):
    reference=Path(reference)/'ordinary';captured=Path(captured)/'ordinary'
    plans=[read(p/'plan.json') for p in [reference,captured]]
    if plans[0]['cases']!=plans[1]['cases']:raise ValueError('different cases/configurations')
    protocols=[{k:v for k,v in p['protocol'].items() if k!='motion_capture'} for p in plans]
    if protocols[0]!=protocols[1]:raise ValueError('different evaluation protocols')
    a=plans[0]['protocol']['motion_capture'];b=plans[1]['protocol']['motion_capture']
    if a['capture']!='off' or b['capture']!='on' or {k:v for k,v in a.items() if k!='capture'}!={k:v for k,v in b.items() if k!='capture'}:
        raise ValueError('expected identical adapter/source with capture off/on only')
    rows=[]
    for c in plans[0]['cases']:
        records=[read(p/'trials'/(c['id']+'.json')) for p in [reference,captured]]
        if any(r['status']!='completed' for r in records):raise ValueError('incomplete trials')
        dirs=[p/'attempts'/c['id']/('attempt_%03d'%r['attempt']) for p,r in zip([reference,captured],records)]
        actual=[read(d/'actual_protocol.json') for d in dirs]
        signatures=['state_sha256','input_sha256','noise_sha256']
        if any(actual[0]['first'][k]!=actual[1]['first'][k] for k in signatures):raise ValueError('initial state/input/noise differs')
        metrics=['success','env_steps','chunk_calls','event_replans','guidance_calls']
        if any(records[0]['result'][k]!=records[1]['result'][k] for k in metrics):raise ValueError('outcome/action count differs')
        for name in ['executed_actions.json','queue_trace.json']:
            if read(dirs[0]/name)!=read(dirs[1]/name):raise ValueError('actual actions/queue differs: '+name)
        videos=[r['video_artifacts'] for r in records]
        if any(len(v)!=1 for v in videos):raise ValueError('missing single video')
        video_identical=videos[0][0]['sha256']==videos[1][0]['sha256']
        if video_identical:
            for root,artifacts in zip([reference,captured],videos):
                if hashlib.sha256((root/artifacts[0]['path']).read_bytes()).hexdigest()!=artifacts[0]['sha256']:raise ValueError('video SHA changed')
        else:
            frames=[frame_hash(root/v[0]['path']) for root,v in zip([reference,captured],videos)]
            if frames[0]!=frames[1]:raise ValueError('rendered RGB frames differ')
        motion=read(dirs[1]/'motion/status.json')
        steps=[json.loads(l) for l in (dirs[1]/'motion/steps.jsonl').read_text().splitlines()]
        if motion['status']!='completed' or len(steps)!=records[1]['result']['env_steps'] or not all(r['returned'] and r['step']==i for i,r in enumerate(steps)):
            raise ValueError('motion trace completion/order differs')
        rows.append({'case':c,'steps':len(steps),'identical_initial_signatures':True,'identical_actions_and_queue':True,
            'identical_video_bytes':video_identical,'identical_rendered_video':True,
            'reference_callback_wall_s':records[0]['wall_s'],'capture_callback_wall_s':records[1]['wall_s'],
            'reference_episode_wall_s':records[0]['result']['episode_wall_s'],'capture_episode_wall_s':records[1]['result']['episode_wall_s'],
            'capture_diagnostics':motion,'motion_updates':sum(s['controller_updates'] for s in steps)})
    return {'status':'passed','scope':'same-case capture equivalence on tested rollouts; not a universal zero-overhead guarantee',
        'cases':rows,'gpu_extra_forward':False,'overhead_limit':'one pair, not a repeated timing benchmark'}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('reference',type=Path);p.add_argument('captured',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=compare(a.reference,a.captured)
    if any(a.output.resolve().is_relative_to(p.resolve()) for p in [a.reference,a.captured]):raise ValueError('write report outside source trials')
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'cases':len(result['cases']),'report':str(a.output)}))
