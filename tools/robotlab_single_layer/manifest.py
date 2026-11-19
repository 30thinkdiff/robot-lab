"""Plan 454 episodes on CPU without creating or processing any feature tensors."""
from collections import Counter
import json
from pathlib import Path

from .baseline import DATA_REVISION, POLICY_REVISION, POLICY_WEIGHT, SHAPES, digest, PINNED


def build_manifest(old_manifest, raw_root, connector_root, baseline_protocol):
    old_manifest, baseline_protocol = Path(old_manifest), Path(baseline_protocol)
    old, baseline = json.loads(old_manifest.read_text()), json.loads(baseline_protocol.read_text())
    if baseline.get('training_entry_sha256') != PINNED['corrector_train.py'] or baseline.get('model_type') != 'mlp':
        raise ValueError('Actual historical MLP protocol required, not J or defaults')
    if old.get('dataset') != 'HuggingFaceVLA/libero' or old.get('revision') != DATA_REVISION:
        raise ValueError('Wrong dataset or revision')
    if digest(old_manifest) != baseline.get('data_manifest_sha256'):
        raise ValueError('Original historical data manifest SHA mismatch')
    expected = {r['episode']: (r['frames'], r['split']) for r in baseline['records']}
    actual = {r['episode']: (r['frames'], r['split']) for r in old['records']}
    if len(expected) != len(baseline['records']) or len(actual) != len(old['records']) or expected != actual:
        raise ValueError('Episode/split mismatch or duplicate episode leakage')
    if len(actual) != 454 or Counter(v[1] for v in actual.values()) != {'train': 364, 'validation': 90}:
        raise ValueError('Formal full Object10 requires the actual historical 364/90 split')
    records = [dict(r, raw_path=str(Path(raw_root) / ('episode_' + str(r['episode']))),
                    connector_path=str(Path(connector_root) / ('episode_' + str(r['episode']))))
               for r in sorted(old['records'], key=lambda r: r['episode'])]
    result = dict(format='single_layer_object10_manifest454_v1', status='planned_not_extracted',
        dataset=old['dataset'], revision=DATA_REVISION, policy_revision=POLICY_REVISION,
        policy_weight_sha256=POLICY_WEIGHT, fps=10., k_step=10, anchor_history=3,
        raw_shape=list(SHAPES['block_02']), raw_dtype='float16', pooling='none',
        layer_indices_zero_based=[2, 8], records=records,
        frames=sum(r['frames'] for r in records),
        train_pairs=sum(r['frames'] - 12 for r in records if r['split'] == 'train'),
        validation_pairs=sum(r['frames'] - 12 for r in records if r['split'] == 'validation'),
        two_layer_payload_bytes=sum(r['frames'] for r in records) * 6 * 2**20,
        historical_manifest_sha256=digest(old_manifest),
        baseline_protocol_sha256=digest(baseline_protocol),
        feature_extraction='BF16 frozen encoder, FP16 raw layers; same image/processor/action provenance as historical cache',
        online_encoder='FP32 policy encoder, retaining historical online precision; no separate BF16 J encoder')
    validate_plan(result)
    return result


def validate_plan(plan):
    rows = plan['records']
    if len({r['episode'] for r in rows}) != len(rows):
        raise ValueError('Duplicate episode / train-validation leakage')
    if any(r['split'] not in ('train', 'validation') for r in rows):
        raise ValueError('Unknown split')
    if (plan['revision'], plan['fps'], plan['k_step'], plan['anchor_history']) != (DATA_REVISION, 10., 10, 3):
        raise ValueError('Timebase/anchor/revision differs from baseline')
    if plan['policy_revision'] != POLICY_REVISION or plan['policy_weight_sha256'] != POLICY_WEIGHT:
        raise ValueError('Policy mismatch')
    if plan['raw_shape'] != [2, 1024, 768] or plan['raw_dtype'] != 'float16' or plan['pooling'] != 'none':
        raise ValueError('Reduced resolution, projection or dtype change is forbidden')
    if plan['layer_indices_zero_based'] != [2, 8]:
        raise ValueError('Layer convention changed')
    if any(type(r['episode']) is not int or type(r['frames']) is not int or r['frames'] <= 12 for r in rows):
        raise ValueError('Invalid episode or no valid baseline transition')
    original = json.loads(Path(__file__).with_name('historical_mlp.json').read_text())
    expected = {r['episode']: (r['frames'], r['split']) for r in original['records']}
    actual = {r['episode']: (r['frames'], r['split']) for r in rows}
    if actual != expected:
        raise ValueError('Full 454 episode/frame/split identities differ from historical MLP')


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
