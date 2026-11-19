"""Copy a paused checkpoint into a versioned engineering continuation, never edit it."""
import copy
import json
from pathlib import Path
import shutil

import torch

from .baseline import digest
from .fast_pipeline import tensor_digest
from .train_monitor import write


def migrate(source,destination,workers,acceptance,*,pipeline_kind='mlp_no_history'):
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError('Fresh continuation directory required')
    if workers not in (2,4,8):raise ValueError('Unsupported worker count')
    if pipeline_kind not in ('mlp_no_history','packed_fp16_transport'):raise ValueError('Unknown engineering continuation')
    status=json.loads((source/'training_status.json').read_text())
    if status['status']!='paused':raise ValueError('Source must be safely paused at an epoch boundary')
    old=json.loads((source/'protocol.json').read_text());new=copy.deepcopy(old)
    if (old['model_type'],old['h_window'],old['batch_size'],old['accumulate'])!=('mlp',1,32,16):
        raise ValueError('Only current single-frame MLP, batch32/accumulate16')
    new['loader']['num_workers']=workers
    back=copy.deepcopy(new);back['loader']['num_workers']=old['loader']['num_workers']
    if back!=old:raise ValueError('Unexpected scientific protocol change')
    latest=source/'latest.pt';old_sha=digest(latest)
    state=torch.load(latest,map_location='cpu',weights_only=True)
    if state['protocol']!=old or state['epoch']!=status['epochs_completed']:raise ValueError('Source checkpoint/protocol/status inconsistent')
    invariant={k:v for k,v in state.items() if k!='protocol'};payload_sha=tensor_digest(invariant)
    destination.mkdir(parents=True)
    state['protocol']=new
    temporary=destination/'latest.pt.partial';torch.save(state,temporary);temporary.rename(destination/'latest.pt')
    reloaded=torch.load(destination/'latest.pt',map_location='cpu',weights_only=True)
    if tensor_digest({k:v for k,v in reloaded.items() if k!='protocol'})!=payload_sha:
        raise ValueError('Checkpoint tensor/optimizer/scheduler/RNG/history payload changed')
    write(destination/'protocol.json',new)
    copied={}
    for name in ('config.json','split_indices.npz','history.json','metrics.csv','initial_validation.json'):
        if (source/name).exists():
            shutil.copyfile(source/name,destination/name);copied[name]=digest(destination/name)
            if copied[name]!=digest(source/name):raise ValueError('Copy changed '+name)
    if (source/'best_model.pt').exists():
        best=torch.load(source/'best_model.pt',map_location='cpu',weights_only=True)
        if best['feature_protocol']!=old:raise ValueError('Best provenance differs')
        best_invariant=tensor_digest({k:v for k,v in best.items() if k!='feature_protocol'})
        best['feature_protocol']=new;torch.save(best,destination/'best_model.pt')
        check=torch.load(destination/'best_model.pt',map_location='cpu',weights_only=True)
        if tensor_digest({k:v for k,v in check.items() if k!='feature_protocol'})!=best_invariant:raise ValueError('Best weights changed')
    if digest(latest)!=old_sha:raise ValueError('Original checkpoint changed')
    record=dict(scope='Engineering continuation; original checkpoint/source/history retained read-only',
        source=str(source),original_latest_sha256=old_sha,new_latest_sha256=digest(destination/'latest.pt'),
        checkpoint_payload_except_protocol_sha256=payload_sha,checkpoint_payload_exact=True,
        inherited_epoch=state['epoch'],inherited_optimizer_steps=state['step'],
        original_protocol_sha256=digest(source/'protocol.json'),new_protocol_sha256=digest(destination/'protocol.json'),
        changes={'loader.num_workers':{'before':old['loader']['num_workers'],'after':workers},
                 'pipeline_kind':pipeline_kind,
                 'collate':('Lossless stored FP16 transport; restore FP32 current/future before target subtraction on GPU'
                            if pipeline_kind=='packed_fp16_transport' else
                            'MLP-only omit unused z_hist; batch/target tensor equality accepted separately')},
        immutable_other_protocol_fields_equal=True,copied_files_sha256=copied,acceptance=str(acceptance))
    write(destination/'ENGINEERING_CONTINUATION.json',record)
    return record
