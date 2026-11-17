"""Read-only robotlab_motion_v1 adapter; no reconstruction of absent states."""
import hashlib
import html
import json
import math


def load_motion(source, base, record, n, fps):
    receipt = (record.get('result') or {}).get('motion_capture')
    missing = {'available': False, 'message': '本回合未记录运动状态；不从动作或视频推造实际位姿。'}
    if not receipt or receipt.get('capture') == 'disabled_control':
        return missing
    try:
        if receipt.get('schema') != 'robotlab_motion_v1':
            raise ValueError('不支持的运动协议版本')
        hashes = receipt.get('files_sha256', {})
        def read(name):
            member = 'motion/' + name
            expected = hashes.get(member)
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError('缺少运动文件SHA绑定')
            raw = source.read(base + member)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('运动文件SHA不匹配')
            return raw
        protocol = json.loads(read('protocol.json'))
        status = json.loads(read('status.json'))
        rows = [json.loads(line) for line in read('steps.jsonl').splitlines() if line.strip()]
        if protocol.get('schema') != receipt['schema'] or status.get('status') != 'completed':
            raise ValueError('运动采集未完整完成')
        case = record['case']
        bound = protocol.get('provenance', {}).get('case', {})
        if any(bound.get(k) != case.get(k) for k in ['id', 'condition', 'seed', 'method']):
            raise ValueError('运动协议与本回合不匹配')
        if not fps or len(rows) != n or any(row.get('step') != i or row.get('returned') is not True for i, row in enumerate(rows)):
            raise ValueError('运动步号/控制频率/完整性不一致')
        chunks = {}
        for row in rows:
            for phase in ['before', 'after']:
                state = row[phase]
                for key, length in [('eef_position_m', 3), ('joint_position_rad', 7)]:
                    values = state[key]
                    if len(values) != length or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                        raise ValueError('运动状态形状或数值异常')
            dt = row['after']['sim_time_s'] - row['before']['sim_time_s']
            if not math.isclose(dt, 1 / fps, abs_tol=1e-6):
                raise ValueError('运动状态时间间隔与控制频率不匹配')
            key = row['chunk_id']
            if key not in chunks:
                chunk = json.loads(read(f'chunks/{key:04d}.json'))
                if chunk.get('chunk_id') != key:
                    raise ValueError('动作块编号不匹配')
                chunks[key] = chunk
            chunk = chunks[key]
            slot = row['slot']
            if not isinstance(slot, int) or not 0 <= slot < len(chunk['nominal']):
                raise ValueError('动作块slot越界')
            row['nominal'] = chunk['nominal'][slot]
            row['nominal_role'] = chunk['role']
        return {'available': True, 'schema': receipt['schema'], 'verified': True,
                'message': '运动文件SHA、回合绑定、步号和50ms状态间隔已核验。',
                'phase': '视频step n对应before；目标、名义端点与after对应本步执行后。末次after无视频帧。',
                'nominal_limit': '名义路径由动作块与OSC增量规则解释，不是VLA预测的未来状态；OGG回合是已引导的最终动作块。',
                'protocol': protocol, 'status': status, 'rows': rows, 'files_sha256': hashes}
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        return {**missing, 'message': '运动数据核验失败，未绑定时间轴：' + str(exc)}


def motion_chart_svg(detail, step):
    """Portable derived figure: actual OSC tracking vs nominal block deviation."""
    rows=detail['motion']['rows'];n=max(1,len(rows)-1)
    values=[r.get(key,{}).get('position_m') for r in rows for key in ['tracking_error','nominal_block_error']]
    maximum=max([v*1000 for v in values if isinstance(v,(int,float)) and math.isfinite(v)]+[.001])*1.1
    x=lambda v:65+v/n*950
    y=lambda v:265-v/maximum*205
    elements=[]
    for i in range(5):
        value=maximum*i/4;yy=y(value)
        elements.append(f'<path d="M65 {yy} H1015" stroke="#e5ebf1"/><text x="60" y="{yy+4}" text-anchor="end">{value:.1f}</text>')
    for key,color in [('tracking_error','#189a90'),('nominal_block_error','#d69232')]:
        path='';connected=False
        for r in rows:
            v=r.get(key,{}).get('position_m')
            if not isinstance(v,(int,float)) or not math.isfinite(v):connected=False;continue
            path+=('L' if connected else 'M')+f'{x(r["step"]):.2f},{y(v*1000):.2f} ';connected=True
        elements.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
    if 0<=step<len(rows):elements.append(f'<path d="M{x(step)} 55 V265" stroke="#1e4058" stroke-dasharray="4 4"/>')
    r=detail['run'];label=html.escape(f"{r['method']} / {r['model']} / {r['condition']} / seed {r['seed']} / step {step}")
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="320" viewBox="0 0 1080 320">'
        '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12" fill="#50677c">'
        f'<text x="65" y="20">{label}</text><text x="65" y="42">Position error / mm: OSC goal (teal); block nominal endpoint (amber)</text>'
        +''.join(elements)+f'<text x="65" y="287">step 0</text><text x="1015" y="287" text-anchor="end">step {len(rows)-1}</text>'
        '<text x="65" y="309">After-action error; video frame n shows BEFORE action n. Nominal path is not a predicted physical state.</text></g></svg>')
