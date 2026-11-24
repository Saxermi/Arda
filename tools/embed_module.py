"""Generate a C string literal header from a UTF-8 Arda standard module."""
import argparse
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def render(name):
    # Octal escapes preserve UTF-8 bytes in C11, unlike JSON's Unicode escapes.
    def escape(data):
        return ''.join({10: r'\n', 13: r'\r', 9: r'\t', 34: r'\"',
                        92: r'\\', 63: r'\?'}.get(b, chr(b) if 32 <= b < 127 else f'\\{b:03o}')
                       for b in data)
    data = (ROOT / 'lib' / f'{name}.arda').read_bytes()
    lines = ''.join(f'"{escape(line)}"\n' for line in data.splitlines(keepends=True)) or '""\n'
    return (f'/* Generated from lib/{name}.arda; do not edit. */\n'
            f'static const char {name}_source[] =\n{lines};\n').encode('ascii')


def main(name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'src' / f'{name}_source.h'
    expected = render(name)
    if args.check:
        if not out.exists() or out.read_bytes() != expected:
            parser.exit(1, f'{out.name} is out of date\n')
        return
    # Atomic replacement; no partially written header is visible to the compiler.
    with tempfile.NamedTemporaryFile(dir=out.parent, delete=False) as tmp:
        tmp.write(expected)
        path = Path(tmp.name)
    path.chmod(0o644)
    path.replace(out)
