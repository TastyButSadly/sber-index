"""Download the pinned public TabPFN-3.5 checkpoint in checked byte ranges."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess
from backtest import ROOT

meta=json.loads((ROOT/'materials/tabpfn35_api.json').read_text(encoding='utf8'))
name='tabpfn-v3.5-20260909.safetensors'
entry=next(f for f in meta['siblings'] if f['rfilename']==name)
size=entry['size']; expected=entry['lfs']['sha256']
folder=ROOT/'models/tabpfn-3.5';folder.mkdir(exist_ok=True)
url=f"https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/{meta['sha']}/{name}"
chunk=64*1024*1024

def fetch(part):
    start=part*chunk;end=min(size,start+chunk)-1
    destination=folder/f'checkpoint_{part:02d}.part'
    if destination.exists() and destination.stat().st_size==end-start+1:return destination
    subprocess.run(['curl.exe','-sS','-L','--fail','--retry','2','--max-time','900',
        '--range',f'{start}-{end}','--max-filesize',str(end-start+1),
        url+f'?download=true&part={part}', '-o',str(destination)],check=True)
    if destination.stat().st_size!=end-start+1:raise ValueError(f'Wrong chunk length: {part}')
    print('Completed chunk',part,flush=True)
    return destination

destination=folder/name
if not destination.exists():
    with ThreadPoolExecutor(max_workers=8) as pool: parts=list(pool.map(fetch,range((size+chunk-1)//chunk)))
    temporary=destination.with_suffix('.assembling')
    digest=hashlib.sha256()
    with temporary.open('wb') as out:
        for part in parts:
            with part.open('rb') as stream:
                while block:=stream.read(4*1024*1024):out.write(block);digest.update(block)
    if digest.hexdigest()!=expected:raise ValueError('Checkpoint SHA-256 mismatch')
    temporary.replace(destination)
    for part in parts:part.unlink()
else:
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=expected:raise ValueError('Existing checkpoint SHA-256 mismatch')
(folder/'manifest.json').write_text(json.dumps({'url':url,'revision':meta['sha'],'bytes':size,'sha256':expected},indent=2),encoding='utf8')
print('Verified full TabPFN-3.5 checkpoint',size,'bytes',flush=True)
