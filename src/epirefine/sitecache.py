"""The site cache's key, a cheap content fingerprint of its inputs, and a lock on its directory.

The manifest (`manifest.json`) holds `inputs`, the key: the assembly and pileup as [path, size, modification time in
whole seconds], the motif list, the motif groups and the rule choices that change which sites are stored; and
`parts`, the Parquet files. Format 2 (this module, from 0.12) adds `format` and `fingerprints`: for the assembly and
the pileup, the SHA-256 of a few fixed blocks (start, middle, end; the whole file when it is small). Path, size and
time alone miss a file replaced by one of the same size within the same second, or copied with its time kept; the
blocks catch any change to the start, middle or end of the file (the header, the sort order, a truncated or
re-written pileup) and cost three reads of 1 MiB, never a read of the whole pileup (308 GB on ZymoFecal).

A manifest without `format` was written by 0.11 or earlier and is format 1. Its key is the same, so it is reused
when the key matches, as before, on path, size and time only (the log says so); `upgrade` adds the fingerprints
without reading more of the pileup than the blocks. Nothing here changes which sites a cache holds.

The lock: a run holds a shared lock on `<cache>/.lock` while it reads the cache, and an exclusive one while it
(re)builds it, so two runs never write one cache at once and a rebuild waits for the runs reading the old cache.
Runs that only read share the lock and do not wait for each other. Where no lock can be taken (no fcntl, a
read-only directory) the run goes on without one and says so.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

FORMAT = 2
BLOCK = 1 << 20  # bytes per fingerprint block
WHOLE_BELOW = 16 << 20  # files up to this size are hashed whole
FINGERPRINTED = ("assembly", "pileup")

try:  # not on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def fingerprint(path) -> dict:
    """Size and SHA-256 of fixed blocks of the file: the whole file up to 16 MiB, else 1 MiB at the start, in the
    middle (at a multiple of the block size) and at the end."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        if size <= WHOLE_BELOW:
            offsets = "all"
            for buf in iter(lambda: fh.read(BLOCK), b""):
                h.update(buf)
        else:
            offsets = [0, (size // 2) // BLOCK * BLOCK, size - BLOCK]
            for off in offsets:
                fh.seek(off)
                h.update(fh.read(BLOCK))
    return {"size": size, "blocks": offsets, "block_size": BLOCK, "sha256": h.hexdigest()}


def read_manifest(cache) -> dict | None:
    try:
        return json.loads((Path(cache) / "manifest.json").read_text())
    except (OSError, ValueError):
        return None


def manifest_format(cached: dict) -> int:
    return int(cached.get("format", 1))


def write_manifest(cache, key: dict, parts: list[str], paths: dict) -> None:
    """A format-2 manifest, written atomically (a reader never sees half a file). paths: input name -> file, for
    the fingerprints."""
    from . import __version__
    man = {"format": FORMAT, "written_by": __version__, "inputs": key, "parts": parts,
           "fingerprints": {k: fingerprint(paths[k]) for k in FINGERPRINTED}}
    _atomic_write(Path(cache) / "manifest.json", json.dumps(man))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _key_diff(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def status(cache, key: dict, paths: dict) -> tuple[bool, str]:
    """Whether a run with this key reuses the cache, and why (one phrase for the log). paths: input name ->
    file (assembly, pileup), read only for the fingerprint blocks of a format-2 manifest."""
    cache = Path(cache)
    cached = read_manifest(cache)
    if cached is None:
        return False, "no site cache (no readable manifest)"
    fmt = manifest_format(cached)
    if fmt > FORMAT:
        from .io import InputError
        raise InputError(f"site cache {cache} has format {fmt}, written by a newer epirefine "
                         f"({cached.get('written_by', '?')}); this version reads up to format {FORMAT}. Use another "
                         f"--cache-dir (rebuilding here would overwrite it).")
    diff = _key_diff(cached.get("inputs", {}), key)
    if diff:
        return False, "inputs differ (" + ", ".join(diff) + ")"
    missing = [f for f in cached.get("parts", []) if not (cache / "sites" / Path(f).name).exists()]
    if missing:
        return False, f"{len(missing)} of {len(cached['parts'])} part files missing"
    if fmt == 1:
        return True, "format 1 (written before 0.12; no content fingerprint): reused on path, size and time"
    for k, fp in cached.get("fingerprints", {}).items():
        if k in paths and fingerprint(paths[k]) != fp:
            return False, f"the {k} has changed (same path, size and time, other content)"
    return True, f"format {fmt}, key and fingerprints match"


def upgrade(cache, paths: dict) -> str:
    """Format 1 -> 2 in place: adds the fingerprints of the given files when the manifest's path, size and time
    of each match them. Reads the fingerprint blocks only. Takes the cache's exclusive lock. Returns what it did."""
    cache = Path(cache)
    with CacheLock(cache) as lock:
        lock.exclusive()
        cached = read_manifest(cache)
        if cached is None:
            return "no manifest: nothing to upgrade"
        if manifest_format(cached) >= FORMAT:
            return f"already format {manifest_format(cached)}"
        from .pipeline import _manifest
        for k in FINGERPRINTED:
            if cached["inputs"].get(k) != _manifest(**{k: paths[k]})[k]:
                return f"not upgraded: the {k} given is not the one the cache was built from ({paths[k]})"
        cached.update(format=FORMAT, fingerprints={k: fingerprint(paths[k]) for k in FINGERPRINTED})
        cached.setdefault("written_by", "0.11 or earlier (format 1, upgraded)")
        _atomic_write(cache / "manifest.json", json.dumps(cached))
        return f"upgraded to format {FORMAT}"


class CacheLock:
    """flock on <cache>/.lock: shared() for reading, exclusive() for writing (waits for the readers), shared()
    again after a write. Released by release() or at the end of the with block."""

    def __init__(self, cache, log=None):
        self.path = Path(cache) / ".lock"
        self.log = log
        self.fh = None
        self.mode = None

    def _open(self) -> bool:
        if self.fh is not None:
            return True
        if fcntl is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.path, "a+")
        except OSError as e:
            if self.log:
                self.log(f"WARNING: no lock on the site cache {self.path.parent} ({e.strerror or e}); make sure no "
                         f"other run writes it at the same time")
            return False
        return True

    def _take(self, mode, what: str) -> None:
        if not self._open():
            return
        try:
            fcntl.flock(self.fh, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            if self.log:
                self.log(f"waiting for another run that {what} the site cache {self.path.parent}")
            fcntl.flock(self.fh, mode)
        self.mode = mode

    def shared(self) -> None:
        """For reading. From exclusive, this lets other readers in (the write is finished)."""
        self._take(fcntl.LOCK_SH if fcntl else 0, "writes")

    def exclusive(self) -> None:
        """For writing. Taken from shared by first letting go: a run that waits for readers must not hold its own
        shared lock (two runs upgrading at once would wait for each other forever)."""
        if self.fh is not None and self.mode == getattr(fcntl, "LOCK_SH", None):
            fcntl.flock(self.fh, fcntl.LOCK_UN)
        self._take(fcntl.LOCK_EX if fcntl else 0, "reads or writes")

    def try_exclusive(self) -> bool:
        """Exclusive without waiting; False when another run holds the lock."""
        if not self._open():
            return True
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self.mode = fcntl.LOCK_EX
        return True

    def release(self) -> None:
        if self.fh is not None:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_UN)
            finally:
                self.fh.close()
                self.fh, self.mode = None, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
