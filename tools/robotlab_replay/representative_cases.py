"""Select diagnostic outcomes, never estimate a population success rate."""
from collections import defaultdict
import hashlib
import json

FOUR = ('j_lvm', 'j_ogg', 'mlp_lvm', 'mlp_ogg')
SIX = FOUR + ('H50', 'H20')


def select_groups(records, conditions, methods=FOUR):
    """One real example per observed outcome, then cover missing perturbations.

    Only completed, fully matched groups qualify. Missing/unknown is not failure.
    Deterministic ordering makes the selection reproducible without seed search.
    """
    groups = defaultdict(dict)
    for r in records:
        c = r['case']
        if r['status'] != 'completed' or c['method'] not in methods:
            continue
        if type(r['result'].get('success')) is not bool:
            raise ValueError('completed outcome is not boolean')
        g = groups[c['condition'], c['seed']]
        if c['method'] in g:
            raise ValueError('duplicate method/condition/seed')
        g[c['method']] = r
    candidates = []
    for (condition, seed), group in sorted(groups.items()):
        if not all(m in group for m in methods):
            continue
        candidates.append({'condition': condition, 'seed': seed,
            'pattern': ''.join(str(int(group[m]['result']['success'])) for m in methods),
            'category': conditions.get(condition, {}).get('category', 'unclassified'),
            'case_ids': [group[m]['case']['id'] for m in methods]})
    selected = []; patterns = set(); categories = set()
    for g in candidates:
        if g['pattern'] not in patterns:
            selected.append({**g, 'reason': 'observed outcome pattern'})
            patterns.add(g['pattern']); categories.add(g['category'])
    for g in candidates:
        if g['category'] not in categories:
            selected.append({**g, 'reason': 'previously unrepresented perturbation category'})
            categories.add(g['category'])
    return selected


def comparison_groups(catalog):
    """Explicit reviewed cross-source joins; never join arbitrary cohorts silently.

    A local manifest references existing original trial bytes. Each reference is
    checked against its SHA, source study, condition, seed and completed outcome.
    No copied trial, altered protocol or invented action/video is introduced.
    """
    output = []
    for path in sorted((catalog.state / 'comparison_sets').glob('*.json')):
        try:
            pending = []
            manifest = json.loads(path.read_text())
            if manifest['schema'] != 'reviewed_replay_comparisons_v1':
                raise ValueError('unknown comparison schema')
            evidence_path = (path.parent / manifest['evidence_file']).resolve()
            evidence_path.relative_to(path.parent.resolve())
            evidence = evidence_path.read_bytes()
            if hashlib.sha256(evidence).hexdigest() != manifest['evidence_sha256']:
                raise ValueError('comparison evidence SHA mismatch')
            proof=json.loads(evidence)
            if proof.get('status') != 'passed':
                raise ValueError('historical bridge not passed')
            for group in manifest['groups']:
                methods = {};initial_state=None
                for method, ref in group['methods'].items():
                    r = catalog.runs[ref['id']]
                    if (r['condition'], r['seed']) != (group['condition'], group['seed']):
                        raise ValueError('condition or seed mismatch')
                    if r['forced'] or r['status'] != 'completed' or type(r['success']) is not bool:
                        raise ValueError('requires natural completed trials')
                    source_study = r['_protocol'].get('study_id', r['experiment'])
                    if source_study != ref['source_study']:
                        raise ValueError('source cohort mismatch')
                    raw = r['_source'].read(r['member'])
                    if hashlib.sha256(raw).hexdigest() != ref['trial_sha256']:
                        raise ValueError('source trial SHA mismatch')
                    if method != ref['method'] or r['method'] != ref['recorded_method']:
                        raise ValueError('method mismatch')
                    if manifest.get('comparison_kind')=='ogg_loss_modes':
                        from loss_casebank import validate_reference
                        validate_reference(method,ref,r)
                    elif method not in FOUR or (method.startswith('j_') and r['method'] != method) or (
                        method.startswith('mlp_') and (r['model'] != 'MLP' or r['method'] not in [method, method[4:]])):
                        raise ValueError('model/method alias mismatch')
                    methods[method] = r
                if manifest.get('comparison_kind')=='ogg_loss_modes':
                    from loss_casebank import validate_group
                    initial_state=validate_group(methods,group,proof)
                pending.append({'study': manifest['study'], 'condition': group['condition'],
                    'seed': group['seed'], 'phase': 'loss_comparison' if manifest.get('comparison_kind')=='ogg_loss_modes' else '桥接审查·跨批次同条件诊断',
                    'pairing_note': manifest['pairing_note'], 'initial_state_sha256':initial_state, 'runs': methods})
            output.extend(pending)
        except (OSError, ValueError, KeyError, TypeError) as e:
            catalog.errors.append(path.name + ': 对照清单未通过 ' + str(e))
    return output
