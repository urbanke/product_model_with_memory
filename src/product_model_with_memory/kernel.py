"""Compiled saddle solve, built on first use.

`_kernel.c` is compiled into a shared library the first time it is
needed and cached beside the source, keyed on a hash of the source and
the interpreter's platform tag, so a changed kernel rebuilds itself and
two platforms sharing a checkout do not collide.  If no compiler is
available, or the build fails, `solve_peak` is None and every caller
keeps the Python path; nothing depends on this being present.

The flags matter.  `-ffp-contract=off` forbids fusing a*b+c into a
single rounding step, which otherwise moves results in the last bits;
`-fno-fast-math` is stated rather than assumed.  Do not add
`-ffast-math`.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

_SRC = Path(__file__).with_name("_kernel.c")

CFLAGS = ["-O3", "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared"]


def _library_path() -> Path:
    digest = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]
    tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return _SRC.with_name(f"_kernel_{tag}_{digest}.so")


def _build() -> Path | None:
    lib = _library_path()
    if lib.exists():
        return lib
    cc = os.environ.get("CC") or ("clang" if sys.platform == "darwin" else "cc")
    # build to a temporary name in the same directory, then rename, so
    # that concurrent workers cannot observe a half-written library
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=str(lib.parent))
    os.close(fd)
    try:
        subprocess.run([cc, *CFLAGS, "-o", tmp, str(_SRC), "-lm"],
                       check=True, capture_output=True)
        os.replace(tmp, lib)
    except (OSError, subprocess.CalledProcessError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    return lib


def _load():
    if os.environ.get("PMM_KERNEL", "1").lower() in ("0", "no", "false"):
        return None
    lib = _build()
    if lib is None:
        return None
    try:
        dll = ctypes.CDLL(str(lib))
    except OSError:
        return None
    fn = dll.pmm_solve_peak
    P = ctypes.c_void_p
    fn.argtypes = [P, P, P, P, P, P, P, ctypes.c_long, P, ctypes.c_long,
                   ctypes.c_double, ctypes.c_double, ctypes.c_double,
                   ctypes.c_double, ctypes.c_double, P, P, P, P]
    fn.restype = ctypes.c_int

    interp = dll.pmm_interp_column
    interp.argtypes = [P, ctypes.c_long, ctypes.c_long, P, ctypes.c_long,
                       ctypes.c_double, ctypes.c_double, P, P, P, P, P, P, P]
    interp.restype = None
    return fn, interp


_loaded = _load()
solve_peak, interp_column = _loaded if _loaded else (None, None)
available = solve_peak is not None
