#!/usr/bin/env python3
"""Memoryless baseline over a STANDARD LLM tokenizer (GPT-2 / cl100k).

Motivation: the long-term goal is a language model built on this
construction, and language models are trained on subword vocabularies,
not on our adaptive word/byte segmentation.  Measuring our estimator on
the very token stream an LLM would see makes the two lines of work
directly comparable, and the pipeline is far simpler than our own
tokenizer: a fixed vocabulary means no escape symbol, no spelling
stream, no case stream --- one stream, one alphabet, one number.

    pip install tiktoken
    python scripts/llm_token_baseline.py --file data/enwik8 \
        --encoding gpt2 --out output/llm_enwik8 --jobs 12

BENCHMARK CAVEAT, stated in the output and in results.json: a
pretrained BPE vocabulary is DATA-DERIVED and EXTERNAL.  Under the
benchmark rules it would have to be shipped with the decompressor and
counted (the vocabulary file is ~0.5-1 MB, i.e. ~0.04 bits/character on
a 10^8-byte file and ~0.004 on 10^9), and it was trained on web text
that may overlap Wikipedia, which is the kind of prior exposure the
rules exist to exclude.  These runs are therefore reported as an
LLM-comparability line, NOT as a benchmark entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
)

# tiktoken fetches its vocabulary over the network on first use.  Where
# that host is unreachable (a firewalled network), the files can be
# obtained by any other means and dropped into a cache directory: the
# file name is the SHA-1 of the URL, which is exactly how tiktoken
# addresses its own cache, so a correctly named file is indistinguishable
# from one it downloaded itself.  --vocab-dir points at that directory,
# --show-vocab-urls prints the plan and exits.
BLOB_URLS = {
    "gpt2": ("https://openaipublic.blob.core.windows.net"
             "/gpt-2/encodings/main/vocab.bpe",
             "https://openaipublic.blob.core.windows.net"
             "/gpt-2/encodings/main/encoder.json"),
    "r50k_base": ("https://openaipublic.blob.core.windows.net"
                  "/encodings/r50k_base.tiktoken",),
    "p50k_base": ("https://openaipublic.blob.core.windows.net"
                  "/encodings/p50k_base.tiktoken",),
    "cl100k_base": ("https://openaipublic.blob.core.windows.net"
                    "/encodings/cl100k_base.tiktoken",),
    "o200k_base": ("https://openaipublic.blob.core.windows.net"
                   "/encodings/o200k_base.tiktoken",),
}


def cache_key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def show_vocab_urls(encoding: str, vocab_dir: str | None) -> None:
    urls = BLOB_URLS.get(encoding)
    if not urls:
        raise SystemExit(f"no download plan known for encoding {encoding!r}; "
                         f"known: {', '.join(sorted(BLOB_URLS))}")
    where = vocab_dir or "vocab_cache"
    print(f"encoding {encoding!r} needs {len(urls)} file(s).\n"
          f"Fetch each URL by whatever means works on this network, save it\n"
          f"in {where}/ under the name shown, then rerun with "
          f"--vocab-dir {where}\n")
    for u in urls:
        print(f"  {cache_key(u)}\n      <- {u}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file")
    p.add_argument("--out")
    p.add_argument("--encoding", default="gpt2",
                   help="tiktoken encoding name: gpt2, r50k_base, "
                        "cl100k_base, o200k_base")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--bytes", type=int, default=None)
    p.add_argument("--vocab-dir", default=None,
                   help="directory holding pre-fetched vocabulary files "
                        "(sets TIKTOKEN_CACHE_DIR); use when this machine "
                        "cannot reach the download host")
    p.add_argument("--show-vocab-urls", action="store_true",
                   help="print what to download and under what name, "
                        "then exit")
    args = p.parse_args()

    if args.show_vocab_urls:
        show_vocab_urls(args.encoding, args.vocab_dir)
        return
    if not args.file or not args.out:
        raise SystemExit("--file and --out are required")

    if args.vocab_dir:
        d = Path(args.vocab_dir).expanduser().resolve()
        if not d.is_dir():
            raise SystemExit(f"--vocab-dir {d} is not a directory")
        os.environ["TIKTOKEN_CACHE_DIR"] = str(d)
        missing = [u for u in BLOB_URLS.get(args.encoding, ())
                   if not (d / cache_key(u)).exists()]
        if missing:
            print(f"WARNING: {len(missing)} expected file(s) not in {d}; "
                  f"run --show-vocab-urls for the list", flush=True)

    try:
        import tiktoken
    except ImportError:
        raise SystemExit("needs tiktoken:  pip install tiktoken")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    raw = Path(args.file).read_bytes()
    if args.bytes:
        raw = raw[: args.bytes]
    n = len(raw)

    try:
        enc = tiktoken.get_encoding(args.encoding)
    except Exception as exc:                       # network or cache miss
        raise SystemExit(
            f"could not load the {args.encoding!r} vocabulary: {exc}\n\n"
            f"If this machine cannot reach the download host, fetch the "
            f"files elsewhere and point --vocab-dir at them:\n"
            f"    python {Path(__file__).name} --encoding {args.encoding} "
            f"--show-vocab-urls\n")
    text = raw.decode("utf-8", errors="surrogateescape")
    ids = enc.encode(text, disallowed_special=())
    print(f"{args.file}: {n:,} bytes -> {len(ids):,} tokens "
          f"({n/len(ids):.2f} bytes/token), vocabulary {enc.n_vocab:,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # exactness: the stream must reconstruct the file byte for byte
    back = enc.decode(ids).encode("utf-8", errors="surrogateescape")
    if back != raw:
        raise SystemExit("ROUND TRIP FAILED: this encoding is not exactly "
                         "invertible on this file")
    print(f"  round trip verified ({time.time()-t0:.0f}s)", flush=True)

    ids = np.asarray(ids, dtype=np.int64)
    counts = np.bincount(ids, minlength=enc.n_vocab)
    profile = tuple(sorted((int(c) for c in counts if c > 0), reverse=True))
    d = enc.n_vocab
    l_max = default_l_max(d)
    types = len(profile)
    freq = counts[counts > 0] / len(ids)
    h0 = float(-(freq * np.log2(freq)).sum())
    print(f"  {types:,} distinct tokens used, order-0 entropy "
          f"{h0:.4f} bits/token, d={d:,}, L<={l_max}", flush=True)

    def progress(evt, _):
        kind, k, total = evt
        if kind == "tables" and (k % 2000 == 0 or k == total):
            print(f"  tables: {k}/{total} ({time.time()-t0:.0f}s)", flush=True)

    res = depth_averaged_codelength_profiles(
        {0: profile}, d=d, l_max=l_max, jobs=args.jobs,
        progress=progress)[0]
    total_bits = -res.log2_q_avg

    payload = {
        "file": args.file, "bytes": n, "encoding": args.encoding,
        "tokens": int(len(ids)), "vocabulary_size": d,
        "distinct_tokens_used": types,
        "bytes_per_token": n / len(ids),
        "order0_entropy_bits_per_token": h0,
        "codelength_bits_per_token": total_bits / len(ids),
        "redundancy_bits_per_token": total_bits / len(ids) - h0,
        "bits_per_character": total_bits / n,
        "total_bits": total_bits,
        "posterior_mode_depth": res.posterior_mode,
        "benchmark_status": (
            "NOT a benchmark entry: the BPE vocabulary is external and "
            "data-derived, so it would have to be shipped and counted "
            "(~0.5-1 MB), and it may have been trained on text "
            "overlapping the test file"),
        "seconds": time.time() - t0,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    print(f"\n  order-0 entropy    : {h0:.4f} bits/token")
    print(f"  layered memoryless : {payload['codelength_bits_per_token']:.4f} "
          f"bits/token   = {payload['bits_per_character']:.4f} bits/character")
    print(f"  redundancy         : {payload['redundancy_bits_per_token']:+.5f} "
          f"bits/token")
    print(f"  posterior-mode L   : {res.posterior_mode}")
    print("  NOTE: LLM-comparability line, not a benchmark entry "
          "(external vocabulary).")
    print(f"written: {out_dir/'results.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
