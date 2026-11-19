"""Read-only live status and an explicit rolling completed-episode ETA."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
import time
from zoneinfo import ZoneInfo


def estimate(total_frames, completed_frames, current_frames, history):
    remaining=max(0,total_frames-completed_frames-current_frames)
    recent=[r for r in history if r.get('wall_seconds',0)>0 and r.get('frames',0)>0][-5:]
    if not recent:
        return dict(remaining_frames=remaining,frames_per_second=None,eta_seconds=None,
                    eta_basis='等待首个完整Episode校验发布；不以13帧探针外推')
    rate=sum(r['frames'] for r in recent)/sum(r['wall_seconds'] for r in recent)
    return dict(remaining_frames=remaining,frames_per_second=rate,eta_seconds=remaining/rate,
                eta_basis='最近%d个已完成Episode端到端耗时（含读写/SHA/发布），仅估算提取，不含训练'%len(recent))


def process_alive(pid):
    if not pid:return False
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return None


def status(root):
    root=Path(root);path=root/'status.json'
    if not path.exists():return dict(status='not_started',directory=str(root))
    value=json.loads(path.read_text())
    value['process_alive']=process_alive(value.get('pid'))
    value['status_age_seconds']=max(0,time.time()-path.stat().st_mtime)
    if value.get('status')=='running' and value['process_alive'] is False:
        value['saved_status']='running';value['status']='process_missing_requires_resume'
    seconds=value.get('eta_seconds')
    if seconds is not None:
        value['eta_hours']=round(seconds/3600,3)
        if value.get('status')=='running':
            value['estimated_finish_beijing']=(datetime.now(ZoneInfo('Asia/Shanghai'))+timedelta(seconds=seconds)).isoformat(timespec='seconds')
    return value


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True,type=Path)
    a=p.parse_args();print(json.dumps(status(a.root),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
