#!/usr/bin/env python3
"""Provide the text8 corpus at data/text8.

Order of preference:
  1. data/text8 already present (verified)
  2. data/text8.gz bundled in the git repo (decompressed locally --
     works offline, e.g. on cluster nodes without internet)
  3. download from the canonical source or its mirror

The result is the canonical 100,000,000-byte file (lowercase a-z and
space), md5 3bea1919949baf155f99411df5fada7e.
"""

import gzip
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

EXPECTED_BYTES = 100_000_000
EXPECTED_MD5 = "3bea1919949baf155f99411df5fada7e"

ZIP_URL = "https://mattmahoney.net/dc/text8.zip"
GZ_URL = "https://github.com/piskvorky/gensim-data/releases/download/text8/text8.gz"


def fetch(url: str) -> bytes:
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url) as r:
        return r.read()


def main() -> int:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    out = data_dir / "text8"
    local_gz = data_dir / "text8.gz"
    data_dir.mkdir(exist_ok=True)
    if out.exists() and out.stat().st_size == EXPECTED_BYTES:
        data = out.read_bytes()
    elif local_gz.exists():
        print(f"decompressing bundled {local_gz} ...")
        data = gzip.decompress(local_gz.read_bytes())
        out.write_bytes(data)
    else:
        try:
            data = zipfile.ZipFile(io.BytesIO(fetch(ZIP_URL))).read("text8")
        except Exception as e:  # noqa: BLE001 - fall back to mirror
            print(f"primary failed ({e}); trying mirror")
            data = gzip.decompress(fetch(GZ_URL))
        out.write_bytes(data)
    ok_size = len(data) == EXPECTED_BYTES
    ok_md5 = hashlib.md5(data).hexdigest() == EXPECTED_MD5
    print(f"size {len(data)} ok={ok_size}; md5 ok={ok_md5}; file: {out}")
    return 0 if (ok_size and ok_md5) else 1


if __name__ == "__main__":
    sys.exit(main())
