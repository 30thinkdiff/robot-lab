"""显式启动的三帧真实示范视觉Hook核验；GPU须空闲，禁止训练/仿真/联网。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def run(args):
    if args.output.exists():raise FileExistsError('独立探针目录已存在，不覆盖')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU仍有进程，禁止提取')
    state=json.loads(Path('/root/robotlab_replay_casebank_20261112_003/results/execution.json').read_text())
    if state['status'] not in ['paused','completed']:raise RuntimeError('评测尚未安全结束')
    if shutil.disk_usage(args.output.parent).free<5*2**30:raise RuntimeError('系统盘不足5GiB，不启动小缓存')
    os.environ.update(HF_HOME=str(args.runtime/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                      LEROBOT_SRC=str(args.runtime/'lerobot/src'))
    sys.path.insert(0,str(args.source_root/'src'))
    # 与既有评测一致：官方冻结LeRobot优先，参考仓库仅提供siglip_dynamics。
    sys.path.insert(0,str(args.runtime/'lerobot/src'))
    import lerobot
    if not Path(lerobot.__file__).resolve().is_relative_to((args.runtime/'lerobot/src').resolve()):raise RuntimeError('LeRobot导入来源不符')
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    from corrector_data_workbench import cached_visual_extractor,cached_policy_snapshot,decode_rgb,IMAGE_KEYS
    from siglip_dynamics.extract import SmolVLAExtractor
    from corrector_multilayer import ReadOnlyVisionTap,write_frame,aligned_pairs
    torch.set_num_threads(4)
    path=args.data_root/'data/chunk-000/file-213.parquet'
    table=pq.read_table(path,columns=['episode_index','frame_index','timestamp',*IMAGE_KEYS])
    rows={r['frame_index']:r for r in table.to_pylist() if r['episode_index']==813 and r['frame_index'] in [0,10,20]}
    if set(rows)!={0,10,20}:raise RuntimeError('固定真实示范帧缺失，不用模拟图像代替')
    snapshot=Path(cached_policy_snapshot(False));weight=snapshot/'model.safetensors';weight_before=digest(weight)
    encoder=cached_visual_extractor(SmolVLAExtractor,snapshot,'cuda',allow_network=False)
    encoder.policy.eval().requires_grad_(False);wrapper=encoder.model.vlm_with_expert;vision=wrapper.get_vlm_model().vision_model
    layers=(2,5,len(vision.encoder.layers)-1)
    if len(vision.encoder.layers)!=12 or layers!=(2,5,11):raise RuntimeError('冻结实际视觉层数与已审查结构不一致')
    cameras=list(encoder.policy.config.image_features)
    if len(cameras)!=2:raise RuntimeError('相机数量异常')
    args.output.mkdir();timings=[];records=[];native_shapes={};peak_increments=[];finite={};equal=[]
    for frame,row in sorted(rows.items()):
        views={}
        for key,camera in zip(IMAGE_KEYS,cameras):
            blob=row[key].get('bytes')
            if not blob:raise RuntimeError('示范图像非内嵌真实字节，不在线下载')
            image=decode_rgb(blob);image_tensor=encoder._prepare_image(image)
            with torch.inference_mode():
                wrapper.embed_image(image_tensor)  # warm-up; 不保留随机测试图。
                for repeat in range(3):
                    torch.cuda.synchronize();start=time.perf_counter();plain=wrapper.embed_image(image_tensor);torch.cuda.synchronize()
                    baseline_s=time.perf_counter()-start;start_allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    with ReadOnlyVisionTap(wrapper,layers) as tap:
                        torch.cuda.synchronize();start=time.perf_counter()
                        with tap.capture(camera) as captured:observed=wrapper.embed_image(image_tensor)
                        torch.cuda.synchronize();hook_s=time.perf_counter()-start
                    peak_increments.append(max(0,torch.cuda.max_memory_allocated()-start_allocated))
                    exact=torch.equal(plain,observed) and torch.equal(observed,captured['connector']);equal.append(exact)
                    if not exact:raise RuntimeError('只读Hook改变原输出或Connector捕获不一致，停止，不放宽阈值')
                    if any(not torch.isfinite(v).all() for v in captured.values()):raise RuntimeError('多层特征包含非有限值')
                    timings.append({'frame':frame,'camera':camera,'repeat':repeat,'plain_s':baseline_s,'hook_s':hook_s,'exact_equal':exact})
                views[camera]=captured;native_shapes[camera]={k:list(v.shape) for k,v in captured.items()};finite[camera]=True
        records.append(write_frame(args.output/'cache',813,frame,float(row['timestamp']),views,cameras))
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>100*2**20:raise RuntimeError('探针缓存超100MiB预算')
        del views
    if digest(weight)!=weight_before:raise RuntimeError('只读权重SHA改变')
    block=vision.encoder.layers[layers[0]]
    if any(len(m._forward_hooks) for m in [*[vision.encoder.layers[i] for i in layers],vision.post_layernorm,wrapper.get_vlm_model().connector]):raise RuntimeError('Hook未清理')
    manifest={'schema':'multilayer_visual_v1','purpose':'predict action-conditioned per-layer future visual change; not task failure classification',
        'policy_revision':snapshot.name,'weight_sha256':weight_before,'weight_sha_unchanged':True,
        'source_parquet_sha256':digest(path),'data_revision':'86958911c0f959db2bbbdb107eb3e17c5f9c798e','episode':813,
        'layer_indices_zero_based':list(layers),'vision_layer_count':len(vision.encoder.layers),'vision_class':type(vision).__name__,
        'vision_config':vision.config.to_dict(),'connector_class':type(wrapper.get_vlm_model().connector).__name__,
        'camera_keys':cameras,'native_shapes':native_shapes,'records':records,'pairs_k10':aligned_pairs(records,10),
        'prediction_seconds':1.,'data_fps':10.,'online_control_hz':20.,'timings':timings,
        'mean_plain_ms':statistics.mean(t['plain_s'] for t in timings)*1000,
        'mean_hook_ms':statistics.mean(t['hook_s'] for t in timings)*1000,
        'max_hook_incremental_peak_mib':max(peak_increments)/2**20,'final_allocated_gib':torch.cuda.memory_allocated()/2**30,
        'all_original_outputs_exact_equal':all(equal),'finite':finite,'hooks_removed':True,
        'cache_uncompressed_bytes':sum(r['uncompressed_bytes'] for r in records),'cache_actual_bytes':sum(r['file_bytes'] for r in records),
        'all_66984_frames_projected_uncompressed_gib':statistics.mean(r['uncompressed_bytes'] for r in records)*66984/2**30,
        'source_files':{p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('corrector_multilayer.py')]},
        'limits':['3 frames from one demonstration, no training or closed-loop claims','spatial mean pooling creates a new explicit feature protocol',
            'intermediate blocks are pre-final-LayerNorm; connector is final representation only','native bfloat16 copied to float16 cache; no normalization rescaling',
            'timing is a small synchronous microbenchmark, not amortized online control latency','Case Bank does not contain these multilayer features']}
    (args.output/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['vision_layer_count','native_shapes','mean_plain_ms','mean_hook_ms','max_hook_incremental_peak_mib','cache_actual_bytes','all_original_outputs_exact_equal','hooks_removed']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime',type=Path,default=Path('/root/autodl-tmp/corrector/runtime'));p.add_argument('--source-root',type=Path,default=Path('/root/autodl-tmp/corrector/runtime/vla-corrector-reference'))
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/corrector/object10-work/source'));a=p.parse_args();run(a)
