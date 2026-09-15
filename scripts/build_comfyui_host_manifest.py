#!/usr/bin/env python3
"""Bind a public profile to exact prehashed installed ComfyUI assets.

This only writes a private manifest. It never starts a runtime, installs models,
changes their contents, or changes GPU settings. Run before installing a binding.
"""
import argparse
import hashlib
import json
from pathlib import Path

from native_task_host import atomic_write
from runtime_contracts import model_set_fingerprint, runtime_profile_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--model-set', type=Path, required=True)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    models = json.loads(args.model_set.read_text())
    if profile['model_set_fingerprint'] != model_set_fingerprint(models):
        raise ValueError('profile/model identity mismatch')
    files = []
    for component in models['components']:
        matches = list((args.comfy_root / 'models').rglob(component['artifact']))
        if len(matches) != 1:
            raise ValueError('model artifact must resolve uniquely: ' + component['artifact'])
        path = matches[0].resolve(strict=True)
        before = path.stat()
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        after = path.stat()
        identity = lambda stat: [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
        if identity(before) != identity(after) or digest != component['sha256']:
            raise ValueError('model asset changed or hash mismatch: ' + component['artifact'])
        files.append({'path': str(path), 'sha256': digest,
                      'file_identity': identity(after), 'role': 'model'})
    manifest = {'runtime_instance': args.instance,
                'profile_name': args.profile.name.removesuffix('.runtime-profile.json'),
                'profile_fingerprint': runtime_profile_fingerprint(profile),
                'model_set': profile['model_set'], 'files': files}
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write(args.output, json.dumps(manifest, indent=2).encode())
    print(json.dumps({'manifest': str(args.output), 'verified_models': len(files),
                      'profile_fingerprint': manifest['profile_fingerprint']}))


if __name__ == '__main__':
    main()
