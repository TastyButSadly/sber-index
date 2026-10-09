"""Restore frozen experiment caches; reject archive paths outside the repository."""
import zipfile
from backtest import ROOT


def main():
    for path in sorted((ROOT/'artifacts').glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                target = (ROOT/entry.filename).resolve()
                if not target.is_relative_to(ROOT.resolve()): raise ValueError('Unsafe archive entry')
            archive.extractall(ROOT)
    print('Experiment caches restored.')


if __name__ == '__main__':
    main()
