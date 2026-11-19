"""Full-epoch disposable certification; official checkpoint stays unchanged."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from .baseline import load_reference,digest
from .execution_bench import run_case,exact,prefetch_rng_probe
from .train_monitor import write


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('base','manifest','checkpoint','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',phase='preflight');write(a.output/'status.json',state)
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
        torch.set_num_threads(4)
        ref=load_reference(a.base/'night-code-1e80242789',a.base/'runtime/vla-corrector-reference')
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True);source_sha=digest(a.checkpoint)
        plan=json.loads(a.manifest.read_text());prefetch_rng_probe()
        rows=[]
        # Candidate first, baseline second: do not give only the candidate the
        # benefit of the previous full baseline's file cache.
        for name,d,pref in [('candidate',True,True),('baseline',False,False)]:
            state.update(phase='full_s2_epoch_and_validation',case=name);write(a.output/'status.json',state)
            r=run_case(ref,plan,saved,'block_02',d,pref,a.output,measure=1530,
                       validation_batches=378,validation_prefetch=False)
            r['name']=name;rows.append(r);write(a.output/(name+'.json'),r)
        exact(rows[0],rows[1])
        if rows[0]['all_updates_parity']['count']!=49447 or rows[0]['validation']['samples']!=12089:
            raise ValueError('Full-epoch sample population differs')
        state.update(phase='s8_phase_policy_parity_and_speed');write(a.output/'status.json',state)
        s8=[]
        for name,d,pref in [('baseline',False,False),('candidate',True,True)]:
            r=run_case(ref,plan,saved,'block_08',d,pref,a.output,measure=96,validation_prefetch=False)
            s8.append(r);write(a.output/('s8_'+name+'.json'),r)
        exact(*s8)
        speed=rows[1]['estimated_epoch_seconds']/rows[0]['estimated_epoch_seconds']
        s8_speed=s8[0]['estimated_epoch_seconds']/s8[1]['estimated_epoch_seconds']
        if digest(a.checkpoint)!=source_sha:raise ValueError('Official source changed')
        result=dict(status='passed' if speed>=1.03 and s8_speed>=1.0 else 'no_robust_gain',
            full_s2_speedup=speed,s8_short_speedup=s8_speed,selected=dict(deferred=True,prefetch=True,validation_prefetch=False),
            full_epoch_counts_exact=True,all_97_updates_parameters_optimizer_rng_metrics_exact=True,
            s8_exact=True,full_iterator_rng_and_tail_exact=True,source_checkpoint_sha256=source_sha,
            inherited_epoch=saved['epoch'],manifest_sha256=digest(a.manifest),
            pipeline_sha256=digest(Path(__file__).with_name('execution_pipeline.py')),
            estimated_epoch_seconds=rows[0]['estimated_epoch_seconds'],
            cases=[{k:v for k,v in r.items() if k not in ('first_update_parity','all_updates_parity','order','validation_order')} for r in rows],
            s8_cases=[{k:v for k,v in r.items() if k not in ('first_update_parity','all_updates_parity','order','validation_order')} for r in s8],
            scope='All engineering updates discarded; full S2 population and short S8 execution check; not new trained model results')
        write(a.output/'result.json',result);state.update(status=result['status'],phase='complete');write(a.output/'status.json',state)
    except BaseException as e:
        state.update(status='error',error=type(e).__name__+': '+str(e));write(a.output/'status.json',state);raise


if __name__=='__main__':main()
