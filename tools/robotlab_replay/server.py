"""RobotLab Replay: Python标准库本地服务，原实验只读，派生导出独立保存。"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import html
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import threading
from urllib.parse import urlparse,parse_qs
import uuid
import webbrowser
import zipfile

from data import Catalog,HERE,clean,finite
from diagnostics import summarize,heatmaps,alignment_steps


def parse_range(value,size):
    if not value:return 0,size-1,False
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
    if not match or not any(match.groups()):raise ValueError('Range不支持')
    a,b=match.groups()
    if not a:
        if int(b)<=0:raise ValueError('Range无效')
        start=max(0,size-int(b));end=size-1
    else:start=int(a);end=min(size-1,int(b)) if b else size-1
    if start<0 or start>=size or end<start:raise ValueError('Range越界')
    return start,end,True


def chart_svg(detail,step):
    width,height=1100,260;pad=45;n=max(1,len(detail['steps'])-1)
    values=[v for row in detail['steps'] for v in [row.get('score'),row.get('executed_score'),row.get('threshold')] if finite(v) and v<=2]
    maximum=max(values+[1])*1.1;elements=[]
    x_at=lambda s:pad+(width-2*pad)*s/n
    start=None;first=detail.get('first_due')
    for row in detail['steps']:
        missing=first is not None and row['step']>=first and row.get('score') is None
        if missing and start is None:start=row['step']
        if not missing and start is not None:
            elements.append(f'<rect x="{x_at(start)}" y="{pad}" width="{x_at(row["step"])-x_at(start)}" height="{height-2*pad}" fill="#fbf0df"/>');start=None
    if start is not None:elements.append(f'<rect x="{x_at(start)}" y="{pad}" width="{max(1,x_at(n)-x_at(start))}" height="{height-2*pad}" fill="#fbf0df"/>')
    for i in range(5):
        y=pad+(height-2*pad)*i/4;value=maximum*(1-i/4)
        elements.append(f'<path d="M45 {y} H1080" stroke="#e2e8f0"/><text x="4" y="{y+4}" font-size="12">{value:.3g}</text>')
    for key,color in [('score','#1879ad'),('threshold','#da8a18'),('executed_score','#0e9a88')]:
        path='';connected=False
        for row in detail['steps']:
            value=row.get(key)
            if not finite(value) or value>maximum:connected=False;continue
            x=pad+(width-2*pad)*row['step']/n;y=height-pad-(height-2*pad)*value/maximum
            path+=('L' if connected else 'M')+f'{x:.2f},{y:.2f} ';connected=True
        elements.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
    high=[r['threshold'] for r in detail['steps'] if finite(r.get('threshold')) and r['threshold']>maximum]
    if high:elements.append(f'<text x="45" y="33" font-size="10" fill="#c18a31">Initial threshold {max(high):.3g} exceeds displayed axis; original values preserved in source.</text>')
    x=pad+(width-2*pad)*max(0,min(step,n))/n;elements.append(f'<path d="M{x} 25 V225" stroke="#17243c" stroke-dasharray="4 4"/>')
    suffix=f" · run already ended at {len(detail['steps'])} steps" if step>=len(detail['steps']) else ' · before run start on manual offset' if step<0 else ''
    label=html.escape(f"{detail['run']['model']} · {detail['run']['method']} · {detail['run']['condition']} · seed{detail['run']['seed']} · requested step{step}"+suffix)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#334155"><text x="45" y="18" font-size="13">{label}</text>'+''.join(elements)+'<text x="45" y="250" font-size="12">Environment step · score blue / threshold amber / executed score teal · missing values are gaps</text></g></svg>'


def export_report(catalog,uids,step,note='',alignment=None):
    if len(uids) not in (1,2):raise ValueError('仅支持单回合或两回合导出')
    details=[catalog.details(uid) for uid in uids]
    if len(details)==2 and (details[0]['run']['condition'],details[0]['run']['seed'])!=(details[1]['run']['condition'],details[1]['run']['seed']):
        raise ValueError('只能导出相同条件/种子的配对')
    local_steps,anchor=alignment_steps(details,step,alignment)
    folder=catalog.state/'exports'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);folder.mkdir(parents=True,exist_ok=False)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'environment_step':step,'note':note[:4000],
        'alignment_basis':'manual event offset; not automatic phase alignment' if anchor else 'environment step, not manipulation phase',
        'manual_alignment':anchor,'environment_steps':local_steps,'sources':[],'files':[]}
    report=['# RobotLab Replay · 事件复盘','',f'环境步：{step}','',note[:4000],'']
    if anchor:report+=['人工事件锚点：'+json.dumps(anchor,ensure_ascii=False),'']
    for i,d in enumerate(details):
        step=local_steps[i]
        label='AB'[i];r=d['run'];metrics=d['metrics'];source={'run_id':r['id'],'model':r['model'],'method':r['method'],
            'condition':r['condition'],'seed':r['seed'],'success':r['success'],'env_steps':r['steps'],
            'provenance':d['source'],'policy_revision':d['protocol'].get('policy_revision'),
            'video_sha256_verified':d['video'].get('sha256_verified',False),'video_alignment':d['video']['alignment'],
            'sync_message':d['video']['message'],'coverage':d['coverage'],'step_available':0<=step<len(d['steps']),
            'selected_environment_step':step,'diagnostic_summary':summarize(d)}
        manifest['sources'].append(source)
        protocol=d['protocol']
        source['frozen_configuration']={k:protocol[k] for k in ['policy_id','policy_revision','fps','control_fps','data_fps',
            'chunk_size','num_steps','prediction_seconds','due_steps','horizon_steps','warmup_steps','eta',
            'guidance_every','guidance_indices','training_seed','threshold'] if k in protocol}
        weights=protocol.get('weights',protocol.get('all_weights',{}))
        if not isinstance(weights,dict):weights={}
        source['checkpoint_hashes']={name:{k:v for k,v in meta.items() if k.endswith('sha256')} for name,meta in weights.items() if isinstance(meta,dict)}
        if isinstance(protocol.get('checkpoint'),dict):source['checkpoint_hashes']['predictor']={k:v for k,v in protocol['checkpoint'].items() if k.endswith('sha256')}
        config=catalog.runs[r['id']]['_record']['case'].get('config',{})
        source['method_configuration']={k:config[k] for k in ['detector','guide','n_action_steps','action_steps','uses_corrector','corrector_sha256'] if k in config}
        if isinstance(protocol.get('calibration'),dict):source['frozen_configuration']['trigger_rule']=protocol['calibration'].get('rule')
        (folder/f'{label}_lvm.svg').write_text(chart_svg(d,step))
        # Exports contain selected numeric fields, no private absolute filesystem paths.
        selected={k:metrics.get(k) for k in ['success','env_steps','chunk_calls','event_replans','guidance_calls','policy_s','monitor_s']}
        report += [f'## {label} · {r["model"]} / {r["method"]}',f'条件：{r["condition"]}；seed={r["seed"]}；当前原始环境步={step}',
            f'指标：{json.dumps(selected,ensure_ascii=False)}',f'同步：{d["video"]["message"]}','']
        with (folder/f'{label}_events.csv').open('w',newline='') as f:
            fields=['environment_step','type','label','step_score','step_threshold','denoise','ogg_loss','guidance_loss',
                    'gradient_l2','clean_action_update_l2','target_norm','discarded_actions']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for e in d['events']:
                row={'environment_step':e['step'],'type':e['kind'],'label':e['label'],
                     'step_score':d['steps'][e['step']].get('score'),'step_threshold':d['steps'][e['step']].get('threshold')}
                row.update({key:e['data'].get(key) for key in fields[5:]});w.writerow(row)
        path=catalog.video_path(catalog.runs[r['id']])
        if 0<=step<len(d['steps']):
            (folder/f'{label}_heatmaps.json').write_text(json.dumps(heatmaps(d,step),ensure_ascii=False,indent=2)+'\n')
        if path and d['video']['alignment']=='recording_verified' and 0<=step<len(d['steps']):
            target=folder/f'{label}_step_{step:04d}.png'
            try:
                subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-vf',f'select=eq(n\\,{step})',
                    '-frames:v','1',str(target)],capture_output=True,timeout=20,check=True)
                source['frame_export']={'frame_index':step,'pts_seconds':d['video']['pts'][step]}
            except (OSError,subprocess.SubprocessError):source['frame_export_error']='ffmpeg不可用或解码失败，未伪造关键帧'
        elif path:source['frame_export_error']='没有精确对齐证据或该回合已结束；未导出标为该step的帧'
    report+=['## 限制','环境步相同不表示相同操作阶段。未记录的Tensor、梯度和机器人测量状态不补造。','原实验只读；本文件和图像为派生导出。']
    (folder/'SUMMARY.md').write_text('\n'.join(report)+'\n')
    for p in folder.iterdir():manifest['files'].append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (folder/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return {'export':folder.name,'files':sorted(p.name for p in folder.iterdir()),'directory':str(folder)}


class Server(ThreadingHTTPServer):
    daemon_threads=True;allow_reuse_address=True
    def __init__(self,address,catalog):self.catalog=catalog;super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,format,*args):pass
    def common(self):
        host=self.headers.get('Host','').split(':')[0]
        if host not in ['localhost','127.0.0.1']:raise ValueError('仅允许本地Host，拒绝DNS rebinding')
    def send(self,value,status=200):
        raw=json.dumps(clean(value),ensure_ascii=False,allow_nan=False).encode();self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store');self.end_headers()
        if self.command!='HEAD':self.wfile.write(raw)
    def file(self,path,mime,video=False):
        size=path.stat().st_size
        try:start,end,partial=parse_range(self.headers.get('Range') if video else None,size)
        except ValueError:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
        self.send_response(206 if partial else 200);self.send_header('Content-Type',mime)
        self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
        if self.command!='HEAD':
            with path.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    data=f.read(min(65536,remaining))
                    if not data:break
                    self.wfile.write(data);remaining-=len(data)
    def do_HEAD(self):self.do_GET()
    def do_GET(self):
        try:
            self.common();parsed=urlparse(self.path);query=parse_qs(parsed.query);catalog=self.server.catalog
            if parsed.path=='/favicon.ico':
                self.send_response(204);self.send_header('Content-Length','0');self.end_headers();return
            if parsed.path=='/api/catalog':self.send(catalog.public());return
            if parsed.path=='/api/run':self.send(catalog.details(query['id'][0]));return
            if parsed.path=='/api/pairs':self.send(catalog.pairs(query['id'][0]));return
            if parsed.path=='/api/diagnostics':
                detail=catalog.details(query['id'][0]);self.send({'summary':summarize(detail),'heatmaps':heatmaps(detail,int(query['step'][0]))});return
            if parsed.path.startswith('/api/video/'):
                uid=parsed.path.rsplit('/',1)[-1];path=catalog.video_path(catalog.runs[uid])
                if path is None:self.send({'error':'未导入视频'},404);return
                self.file(path,'video/webm' if path.suffix.lower()=='.webm' else 'video/mp4',True);return
            if parsed.path.startswith('/api/export-file/'):
                parts=parsed.path.split('/')
                if len(parts)!=5 or not re.fullmatch(r'\d{8}_\d{6}_[0-9a-f]{6}',parts[3]) or not re.fullmatch(r'[A-Za-z0-9_.-]+',parts[4]):raise ValueError('导出路径非法')
                path=(catalog.state/'exports'/parts[3]/parts[4]).resolve();path.relative_to((catalog.state/'exports').resolve())
                if not path.is_file():raise KeyError('文件不存在')
                mime={'.png':'image/png','.svg':'image/svg+xml','.json':'application/json','.csv':'text/csv','.md':'text/plain'}.get(path.suffix,'application/octet-stream')
                self.file(path,mime);return
            static={'/':'index.html','/app.js':'app.js','/diagnostics.js':'diagnostics.js','/style.css':'style.css',
                '/workspace.js':'workspace.js','/workspace_model.mjs':'workspace_model.mjs','/workspace.css':'workspace.css'}
            if parsed.path not in static:self.send({'error':'不存在的地址'},404);return
            mime='text/html; charset=utf-8' if parsed.path=='/' else 'text/css; charset=utf-8' if parsed.path.endswith('.css') else 'text/javascript; charset=utf-8'
            self.file(HERE/'web'/static[parsed.path],mime)
        except (ValueError,KeyError,TypeError) as e:self.send({'error':str(e)},400)
        except (BrokenPipeError,ConnectionResetError):pass
        except (OSError,zipfile.BadZipFile) as e:self.send({'error':'数据读取失败：'+str(e)},422)
    def do_POST(self):
        try:
            self.common()
            if self.headers.get('Origin') not in [None,'http://'+self.headers.get('Host')]:raise ValueError('拒绝跨站写入')
            if self.path!='/api/export':raise ValueError('只允许写入独立派生导出')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<16000:raise ValueError('请求体过大/为空')
            body=json.loads(self.rfile.read(size));self.send(export_report(self.server.catalog,body['ids'],body['step'],str(body.get('note','')),body.get('alignment')))
        except (ValueError,KeyError,TypeError) as e:self.send({'error':str(e)},400)
        except (BrokenPipeError,ConnectionResetError):pass
        except OSError as e:self.send({'error':'独立导出目录不可写或空间不足：'+str(e)},422)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8870)
    p.add_argument('--source',type=Path,action='append',help='可重复：只读ZIP或带plan.json/trials的评测目录；默认自动发现现有五个归档')
    p.add_argument('--state-dir',type=Path,default=HERE/'.local');p.add_argument('--open',action='store_true');args=p.parse_args()
    catalog=Catalog(args.source,args.state_dir);server=Server(('127.0.0.1',args.port),catalog)
    url=f'http://127.0.0.1:{server.server_address[1]}'
    print(f'RobotLab Replay · {len(catalog.runs)} 个真实回合 · {url}\n原数据只读；Ctrl+C关闭服务。',flush=True)
    if args.open:webbrowser.open(url)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:server.server_close();catalog.close()


if __name__=='__main__':main()
