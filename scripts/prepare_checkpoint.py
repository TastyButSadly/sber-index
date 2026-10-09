"""Download the pinned Bolt checkpoint only when missing; verify model SHA-256."""
import hashlib
import json
import urllib.request
from backtest import ROOT


def main():
    folder = ROOT/'models/chronos-bolt-base'; folder.mkdir(parents=True,exist_ok=True)
    meta = json.loads((ROOT/'configs/checkpoint.json').read_text())
    base = 'https://huggingface.co/amazon/chronos-bolt-base/resolve/'+meta['revision']+'/'
    for name in ['config.json','model.safetensors']:
        path = folder/name
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.part')
            urllib.request.urlretrieve(base+name,temporary)
            temporary.replace(path)
    digest = hashlib.sha256()
    with (folder/'model.safetensors').open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
    if digest.hexdigest() != meta['sha256']: raise ValueError('Checkpoint SHA-256 mismatch')
    print('Pinned Chronos-Bolt checkpoint verified.')


if __name__ == '__main__':
    main()
