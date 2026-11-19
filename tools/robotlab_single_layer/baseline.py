"""Load the actual historical MLP implementation, never J training/control."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

DATA_REVISION = '86958911c0f959db2bbbdb107eb3e17c5f9c798e'
POLICY_REVISION = 'a24f1f380461b665158d52acd140152c79e8b36d'
POLICY_WEIGHT = '9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8'
SOURCE_REVISION = '927a06da8f00d7df483459e9039219e076c846de'
PINNED = {
    'corrector_train.py': '8baaf7be9afaccf4bafc9ff05a287da29e80bd578ac6ae28659c1c2c8dcfd95b',
    'corrector_models.py': '811386798aa491d8c0b43c08cd400bf00f397ad380484693689a1912976c4201',
    'corrector_online_runtime.py': '32a6daa67254f6e587e43df0f5c42a0cc83e29fa17854dbab874715f8ae70a5a',
    'corrector_data_workbench.py': 'bb51dfae7c54bd61241c780be8929c056286dca66b142ff0fd95972060c30fe4',
    'corrector_online_practice.py': '5c076c017b086254ec63c3aa8089063d8fbae5840c01c3240f7a57afce99198f',
}
UPSTREAM = {
    'MLP.py': 'b0231797e2308752f2008d8b5399be51dcb669b5db378c1abb4c18f3508e3e77',
    'config.py': 'b664a521512548258cd8c6d69a2863a70ff86919117103429d4f7c4b6a0ac08e',
    'dataset.py': '58d7b5c43eee02b508a7052ab9231f4a0d5246f91b71bdbcc24f8999465ff98d',
    'train.py': 'be24a8d5be163ca3bdf2f5273c1f8802f98d3b2c32f3faacd227b9b2b73ae4f5',
}
SHAPES = {'block_02': (2, 1024, 768), 'block_08': (2, 1024, 768)}
CAMERAS = ['observation.images.camera1', 'observation.images.camera2']
IMAGES = ['observation.images.image', 'observation.images.image2']
TRAINING = dict(epochs=150, patience=5, lr=0.0003, weight_decay=0.0001,
                loss_type='both', beta=1.0, effective_batch=512,
                precision='amp-bfloat16', k_step=10, fps=10.,
                h_window=1, anchor_history=3, seeds=[42, 43, 44])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def historical_record(episode):
    value = json.loads(Path(__file__).with_name('historical_mlp.json').read_text())
    return next(r for r in value['records'] if r['episode'] == episode)


def load_reference(baseline_dir, source_root):
    """Reject wrong copies before the legacy loader could fetch anything."""
    baseline_dir, source_root = Path(baseline_dir).resolve(), Path(source_root).resolve()
    for name, sha in PINNED.items():
        if digest(baseline_dir / name) != sha:
            raise ValueError('Historical baseline source SHA mismatch: ' + name)
    for name, sha in UPSTREAM.items():
        if digest(source_root / 'src/siglip_dynamics' / name) != sha:
            raise ValueError('Upstream source SHA mismatch: ' + name)
    if not (source_root / '.git').exists():
        raise ValueError('Existing verified upstream checkout required; no automatic clone')
    head = subprocess.run(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True, capture_output=True)
    if head.returncode or head.stdout.strip() != SOURCE_REVISION:
        raise ValueError('Existing fixed upstream HEAD required; no fetch/checkout is permitted')
    names = ('corrector_data_workbench', 'corrector_models', 'corrector_train', 'corrector_online_runtime')
    for name in names:
        cached = sys.modules.get(name)
        if cached and Path(cached.__file__).resolve().parent != baseline_dir:
            raise RuntimeError('Another baseline is already imported; use a fresh process: ' + name)
    sys.path.insert(0, str(baseline_dir))
    try:
        modules = {name: importlib.import_module(name) for name in names}
    finally:
        sys.path.pop(0)
    runtime = modules['corrector_online_runtime']
    if digest(inspect.getfile(runtime.guided_velocity)) != PINNED['corrector_online_practice.py']:
        raise ValueError('Resolved OGG implementation differs from the actual historical baseline')
    train = modules['corrector_train']
    config, dataset_cls, model_cls, loss, hashes = train.reference_api(source_root)
    if hashes != UPSTREAM:
        raise ValueError('Reference API source mismatch')
    return dict(train=train, runtime=runtime, config=config, model_cls=model_cls,
                loss=loss, models=modules['corrector_models'], hashes=hashes)


def model_config(reference):
    c = reference['config']
    return c.SiglipMLPConfig(token_dim=768, action_dim=7, action_embed_dim=256,
        scale=c.ModelScale.CUSTOM, custom_widths=(2048, 2048, 2048, 2048),
        dropout=0., rope_theta=10000., ada_rmsnorm_eps=1e-6)


def make_predictor(reference, device='cpu'):
    """Same hidden structure; only input/output token_dim changes to 768."""
    cls = reference['models'].precision_class(reference['model_cls'], TRAINING['precision'])
    return cls(model_config(reference)).to(device)
