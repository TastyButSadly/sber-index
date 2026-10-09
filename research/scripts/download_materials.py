"""Download the public source material and record hashes; preserve existing snapshots."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'materials/contest_page.html': 'https://sber.ru/sberindex/konkurs_sberindex',
    'data/raw/hackathonlicence.zip': 'https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip',
    'data/raw/mo_borders.rar': 'https://www.sberbank.com/common/files/t_dict_municipal.rar',
    'materials/metadata_municipal_dict.pdf': 'https://www.sberbank.ru/common/img/uploaded/files/pdf/sberindex/metadata_municipal_dict_sberindex_2.pdf',
    'data/raw/consumption_current.parquet': 'https://sberindex.ru/api/dataset/v1/download/potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij/parquet',
    'data/raw/mobility_current.parquet': 'https://sberindex.ru/api/dataset/v1/download/indeks-mobilnosti/parquet',
    'materials/chronos_README.md': 'https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/README.md',
    'materials/chronos_quickstart.ipynb': 'https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/notebooks/chronos-2-quickstart.ipynb',
    'materials/chronos_model_api.json': 'https://huggingface.co/api/models/autogluon/chronos-2-small',
    'materials/chronos2_pipeline.py': 'https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/src/chronos/chronos2/pipeline.py',
    'materials/borders.html': 'https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities',
    'materials/data_description.html': 'https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim',
}

def fetch(url, path, insecure_sber=False):
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    cmd = ['curl.exe' if __import__('os').name == 'nt' else 'curl', '-L', '--fail', '--retry', '2', '--max-time', '900']
    if insecure_sber and ('sberbank.' in url or 'sberindex.ru/' in url):
        cmd.append('-k')
    subprocess.run(cmd + [url, '-o', str(temporary)], check=True)
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--insecure-sber', action='store_true', help='Public Sber downloads only: certificate verification fails on this host')
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    sources = SOURCES.copy()
    for filename, url in sources.items():
        if not args.manifest_only:
            fetch(url, ROOT / filename, args.insecure_sber)
    meta = json.loads((ROOT / 'materials/chronos_model_api.json').read_text(encoding='utf-8'))
    rev_file = ROOT / 'models/chronos-2-small/revision.txt'
    revision = rev_file.read_text().strip() if rev_file.exists() else meta['sha']
    for name in ['README.md', 'config.json', 'model.safetensors']:
        file = f'models/chronos-2-small/{name}'
        url = f'https://huggingface.co/autogluon/chronos-2-small/resolve/{revision}/{name}'
        sources[file] = url
        if not args.manifest_only:
            fetch(url, ROOT / file)
    if not args.manifest_only:
        rev_file.write_text(revision, encoding='utf-8')
        with zipfile.ZipFile(ROOT / 'data/raw/hackathonlicence.zip') as archive:
            if archive.testzip():
                raise ValueError('Corrupt dataset archive')
            dest = (ROOT / 'data/raw/hackathon').resolve()
            for name in archive.namelist():
                if not (dest / name).resolve().is_relative_to(dest):
                    raise ValueError('Unsafe archive path')
            archive.extractall(dest)
        subprocess.run(['tar', '-xf', str(ROOT / 'data/raw/mo_borders.rar'), '-C', str(ROOT / 'data/raw')], check=True)
    records = []
    for name, url in sources.items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({'file': name, 'url': url, 'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    inventory = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                 'download_date_moscow': '2026-10-08', 'model_revision': revision,
                 'sber_tls_verification_disabled_on_initial_download': True, 'files': records}
    (ROOT / 'materials/manifest.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(records)} files verified and hashed')

if __name__ == '__main__':
    main()
