"""CPU plan only: call existing unpooled exporter later, never auto-start it."""
from pathlib import Path
import shutil

from .manifest import validate_plan
from .baseline import digest

EXPORTER_SHA256 = '25b988eeec0e7a89220c641a865c1ac017df1086125ec8d5e8151d40306cb28c'


def capacity(plan, raw_root):
    validate_plan(plan)
    root = Path(raw_root)
    parent = root if root.exists() else root.parent
    if not parent.is_dir():
        raise FileNotFoundError('Run capacity check on the destination with an existing target parent; do not infer remote capacity from local /')
    free = shutil.disk_usage(parent).free
    payload = sum(r['frames'] * 6 * 2**20 for r in plan['records'])
    return dict(two_layer_payload_bytes=payload, free_bytes=free,
        reserve_bytes=5 * 2**30, enough_for_fresh_extraction=free >= payload + 5 * 2**30,
        gpu_checked=False, extraction_started=False)


def episode_commands(plan, *, exporter, runtime, source_root, connector_root, data_root, old_manifest):
    """Return arguments, not shell strings/processes. Existing output always stops."""
    validate_plan(plan)
    if digest(exporter) != EXPORTER_SHA256:
        raise ValueError('Use the actual successful unpooled exporter, not the older pooling-only pilot')
    commands = []
    for row in plan['records']:
        target = Path(row['raw_path'])
        if target.exists():
            commands.append(dict(episode=row['episode'], action='verify_existing_before_reuse',
                                 path=str(target), command=None))
            continue
        # The existing exporter creates episode_N under its output. Publish
        # verified output only in the later extractor, never replace raw70.
        attempt = target.parent / '.work' / ('episode_%d_NEW_ATTEMPT' % row['episode'])
        commands.append(dict(episode=row['episode'], action='planned_not_started',
            output=str(attempt), final_path=str(target),
            command=[str(exporter), '--runtime', str(runtime), '--source-root', str(source_root),
                '--deep-root', str(connector_root), '--data-root', str(data_root),
                '--manifest', str(old_manifest), '--output', str(attempt),
                '--episodes', str(row['episode']), '--layers', '2', '8', '--save-fp16',
                '--min-free-gib', '5']))
    return commands
