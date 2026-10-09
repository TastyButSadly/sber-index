"""Verify bundled sources and artifacts against the repository manifest."""
import hashlib
import json
from backtest import ROOT


def main():
    entries = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))['files']
    for entry in entries:
        path = ROOT/entry['path']
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
        if digest.hexdigest() != entry['sha256']: raise ValueError('File changed: '+entry['path'])
    print('Verified',len(entries),'bundled files.')


if __name__ == '__main__':
    main()
