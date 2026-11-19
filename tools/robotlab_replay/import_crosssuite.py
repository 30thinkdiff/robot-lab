"""Import a bounded selection of completed cross-suite cases, never run a GPU.

Default: the two LIBERO-90 task29/init1 seeds, all four frozen-MLP methods.
Reuse the native exporter and importer; verify original trial bytes against the
independently audited public metrics. Old remote/local experiments stay intact.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from corrector_casebank_import import install_bundle
from corrector_crosssuite_plan import METHODS


def selection(metrics):
    import csv
    with Path(metrics).open(newline='') as f:
        rows = [r for r in csv.DictReader(f)
                if r['condition'] == 'libero_90/task_29/init_1' and r['seed'] in ('1000', '1001')]
    if len(rows) != 8 or {(r['seed'], r['method']) for r in rows} != {(str(s), m) for s in (1000, 1001) for m in METHODS}:
        raise ValueError('expected exactly two complete four-method groups')
    if any(r['success'] != 'True' for r in rows if r['method'] == 'mlp_ogg'):
        raise ValueError('selected OGG recovery cases differ from audited results')
    return {r['id']: r['trial_sha256'] for r in rows}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--ssh-host', default='autodl')
    p.add_argument('--ssh-port', type=int)
    p.add_argument('--control-path', type=Path)
    p.add_argument('--state', type=Path, default=HERE / '.local')
    p.add_argument('--metrics', type=Path, default=REPO / 'studies/policy_generalization/results/20261117/crosssuite/trials.csv')
    p.add_argument('--plan', action='store_true')
    a = p.parse_args()
    chosen = selection(a.metrics)
    token = hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()[:16]
    config = {'root': '/root/corrector_crosssuite_20261117_001/transfer',
              'output': '/root/corrector_replay_exports_20261117/crosssuite90_' + token + '.zip',
              'trials': chosen}
    if a.plan:
        print(json.dumps(config, indent=2))
        return
    exporter = base64.b64encode((REPO / 'scripts/corrector_replay_export.py').read_bytes()).decode()
    script = "import base64,hashlib,json,zipfile\nfrom pathlib import Path\n"
    script += 'cfg=json.loads(' + repr(json.dumps(config)) + ')\n'
    script += "root=Path(cfg['root']);target=Path(cfg['output'])\n"
    script += "if json.loads((root/'execution.json').read_text())['status']!='completed':raise ValueError('source evaluation not completed')\n"
    script += "for id,expected in cfg['trials'].items():\n raw=(root/'trials'/(id+'.json')).read_bytes()\n if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('original trial bytes changed')\n"
    script += 'scope={"__name__":"readonly_export_module"}\nexec(compile(base64.b64decode(' + repr(exporter) + '),"<readonly-export>","exec"),scope)\n'
    script += "if not target.exists():\n receipt=scope['export'](root,target,list(cfg['trials']))\n"
    script += "else:\n with zipfile.ZipFile(target) as z:\n  manifest=json.loads(z.read('results/export_manifest.json'))\n  if set(manifest['selected_ids'])!=set(cfg['trials']) or manifest['source_plan_sha256']!=scope['hashlib'].sha256((root/'plan.json').read_bytes()).hexdigest():raise ValueError('existing export differs; preserved')\n  for id,expected in cfg['trials'].items():\n   if hashlib.sha256(z.read('results/trials/'+id+'.json')).hexdigest()!=expected:raise ValueError('existing export trial differs')\n receipt={'completed':8,'bytes':target.stat().st_size,'sha256':scope['hashlib'].sha256(target.read_bytes()).hexdigest()}\n"
    script += "print(json.dumps(dict(receipt,remote_export=str(target),gpu_started=False)))\n"
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    scp = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    if a.ssh_port:
        ssh += ['-p', str(a.ssh_port)]
        scp += ['-P', str(a.ssh_port)]
    if a.control_path:
        ssh += ['-S', str(a.control_path)]
        scp += ['-o', 'ControlPath=' + str(a.control_path)]
    response = subprocess.run(ssh + [a.ssh_host, 'python3 -'], input=script.encode(), capture_output=True)
    if response.returncode:
        raise RuntimeError(response.stderr.decode(errors='replace'))
    receipt = json.loads(response.stdout)
    state = a.state.resolve()
    downloads = state / 'casebank_downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / ('crosssuite90_' + receipt['sha256'][:16] + '.zip')
    if not archive.exists():
        fd, name = tempfile.mkstemp(prefix='crosssuite90_', suffix='.partial', dir=downloads)
        os.close(fd)
        partial = Path(name)
        try:
            subprocess.run(scp + [a.ssh_host + ':' + receipt['remote_export'], str(partial)], check=True)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != receipt['sha256']:
                raise ValueError('download SHA differs')
            os.link(partial, archive)
        finally:
            partial.unlink(missing_ok=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('local archive changed')
    imported = install_bundle(archive, state)
    output = {'selection': config, 'remote_receipt': receipt, 'local_archive': str(archive),
              'import': imported, 'gpu_started': False, 'source_results_modified': False}
    review = state / 'review/crosssuite90'
    review.mkdir(parents=True, exist_ok=True)
    proof = review / ('import_' + receipt['sha256'][:16] + '.json')
    if not proof.exists():
        with proof.open('x') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
