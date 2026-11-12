"""按相同实验、物理条件和seed整理真实成败组合；不混入强制检查。"""
from collections import defaultdict

MODELS={'MLP':'mlp','T-H1':'transformer-h1','T-H3':'transformer-h3','A10-H1':'transformer-a10-h1','A10-H3':'transformer-a10-h3'}
LABELS={'H50':'H50','H20':'H20',**{model+'_'+mode:label+' / '+mode.upper() for label,model in MODELS.items() for mode in ['lvm','ogg']},
        'hybrid_t1_a10_ogg':'T-H1检测 → A10-H1引导','planned_lvm':'A10-H1 / 新规则规划LVM','executed_lvm':'A10-H1 / 新规则已执行LVM'}


def inventory(catalog):
    grouped=defaultdict(dict);phases={}
    for r in catalog.runs.values():
        if r['forced'] or r['status']!='completed':continue
        if r['_protocol'].get('study_id','').startswith('replay_casebank_'):
            study=r['_protocol']['study_id'];method=r['method'];phase=r['_record']['case']['config'].get('phase','未记录')
        elif r['experiment']=='Object10 五模型':
            study='legacy_five_models';phase='历史筛选库'
            if r['method'] in ['H50','H20']:
                if r['model']!='MLP':continue
                method=r['method']
            elif r['model'] in MODELS and r['method'] in ['lvm','ogg']:method=MODELS[r['model']]+'_'+r['method']
            else:continue
        else:continue
        if method not in LABELS:continue
        key=(study,r['condition'],r['seed']);record=r['_record'];base=r['_prefix']+f"attempts/{record['case']['id']}/attempt_{record.get('attempt',1):03d}/";source=r['_source']
        has_trace=source.exists(base+'monitor_steps.jsonl');has_commands=source.exists(base+'controller_commands.json') or source.exists(base+'executed_actions.json')
        item={'id':r['id'],'method':method,'label':LABELS[method],'model':r['model'],'success':r['success'],'steps':r['steps'],
            'video':bool(catalog.video_path(r)),'trace':has_trace,'actions':has_commands,'frame_map':source.exists(base+'frame_map.jsonl')}
        item['replay_complete']=item['video'] and item['actions'] and (has_trace or method in ['H50','H20'])
        grouped[key][method]=item;phases[key]=phase
    rows=[]
    for (study,condition,seed),methods in sorted(grouped.items()):
        rows.append({'study':study,'condition':condition,'seed':seed,'phase':phases[study,condition,seed],
                     'methods':methods,'physical_state':condition})
    return {'groups':rows,'reference_methods':[{ 'id':k,'label':v} for k,v in LABELS.items() if k not in ['H50','H20']],
        'pattern_order':['H50','H20','selected_reference'],'limit':'按结局筛选的诊断案例不是无偏成功率样本；同init不同seed也不是不同物理场景。'}
