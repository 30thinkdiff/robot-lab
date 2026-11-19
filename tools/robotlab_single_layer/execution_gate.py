"""Read-only binding of full real-data execution acceptance to production."""
import json
from pathlib import Path
from .baseline import digest


def require_acceptance(config,manifest):
    r=json.loads(Path(config['receipt']).read_text())
    if digest(config['receipt'])!=config['receipt_sha256']:raise ValueError('Execution receipt changed')
    if (r.get('status')!='passed' or r.get('pipeline_sha256')!=digest(Path(__file__).with_name('execution_pipeline.py'))
        or r.get('manifest_sha256')!=digest(manifest) or r.get('selected')!=config['selected']):
        raise ValueError('Execution acceptance is not bound to this implementation/data/policy')
    for key in ('full_epoch_counts_exact','all_97_updates_parameters_optimizer_rng_metrics_exact',
                's8_exact','full_iterator_rng_and_tail_exact'):
        if r.get(key) is not True:raise ValueError('Execution gate incomplete: '+key)
    if r.get('full_s2_speedup',0)<1.03 or r.get('s8_short_speedup',0)<1:
        raise ValueError('No measured robust speed benefit')

