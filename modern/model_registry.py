"""Local, hash-verified artifact registry. Stores model files; never loads them."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def register_model(source_path: str, name: str, version: str, registry_dir: str = 'model_registry', notes: str = '') -> dict:
    src=Path(source_path)
    if not src.is_file(): raise FileNotFoundError(source_path)
    if not name or any(c in name for c in '/\\') or not version or any(c in version for c in '/\\'):
        raise ValueError('name and version must be non-empty path-safe strings')
    root=Path(registry_dir).resolve(); dest_dir=(root/name/version).resolve()
    if root not in dest_dir.parents: raise ValueError('Invalid registry path')
    dest_dir.mkdir(parents=True,exist_ok=False)
    dest=dest_dir/src.name; shutil.copy2(src,dest)
    metadata={'name':name,'version':version,'artifact':src.name,'sha256':_sha256(dest),'size_bytes':dest.stat().st_size,
              'registered_at_utc':datetime.now(timezone.utc).isoformat(),'notes':notes,'status':'registered_not_validated'}
    (dest_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def verify_model(name: str, version: str, registry_dir: str = 'model_registry') -> dict:
    folder=Path(registry_dir)/name/version
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    artifact=folder/meta['artifact']; actual=_sha256(artifact)
    return {'name':name,'version':version,'valid':actual==meta['sha256'],'expected_sha256':meta['sha256'],'actual_sha256':actual}

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('register'); r.add_argument('artifact'); r.add_argument('--name',required=True); r.add_argument('--version',required=True); r.add_argument('--registry-dir',default='model_registry'); r.add_argument('--notes',default='')
    v=sub.add_parser('verify'); v.add_argument('--name',required=True); v.add_argument('--version',required=True); v.add_argument('--registry-dir',default='model_registry')
    a=p.parse_args(); print(json.dumps(register_model(a.artifact,a.name,a.version,a.registry_dir,a.notes) if a.command=='register' else verify_model(a.name,a.version,a.registry_dir),indent=2))
