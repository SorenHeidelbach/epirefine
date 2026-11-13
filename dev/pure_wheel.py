#!/usr/bin/env python3
"""Derive the pure wheel (py3-none-any) of EpiRefine from one of its compiled wheels (D73).

    python dev/pure_wheel.py <compiled wheel> [--out <directory>]

The pure wheel holds every file of the compiled wheel except the extension module (epirefine/_kernels/_core.*)
and the SBOM of the Rust crates; its METADATA (version, dependencies, entry points, licence files) is the compiled
wheel's, byte for byte. WHEEL says Root-Is-Purelib: true and Tag: py3-none-any; RECORD is written anew. pip and uv
install it only where no compiled wheel of the same version fits (they rank py3-none-any last), and EpiRefine then
runs its numpy code with one warning line. The file is reproducible: entries keep their order and time stamps.
Only the standard library is used.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

DROP = re.compile(r"^epirefine/_kernels/_core[.].*$|^[^/]+\.dist-info/sboms/")


def _hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def pure_wheel(src: Path, out_dir: Path) -> Path:
    m = re.match(r"^(?P<name>[^-]+)-(?P<ver>[^-]+)-.+\.whl$", src.name)
    if not m:
        sys.exit(f"not a wheel file name: {src.name}")
    name, ver = m["name"], m["ver"]
    dist_info = f"{name}-{ver}.dist-info"
    out = out_dir / f"{name}-{ver}-py3-none-any.whl"
    with zipfile.ZipFile(src) as z:
        infos = z.infolist()
        if not any(DROP.match(i.filename) and "/_kernels/" in i.filename for i in infos):
            sys.exit(f"{src.name} has no compiled extension: is it already the pure wheel?")
        files = [(i, z.read(i)) for i in infos
                 if not DROP.match(i.filename) and i.filename not in (f"{dist_info}/RECORD", f"{dist_info}/WHEEL")]
        wheel = z.read(f"{dist_info}/WHEEL").decode()
    keep = [ln for ln in wheel.splitlines() if ln and not ln.startswith(("Root-Is-Purelib:", "Tag:", "Generator:"))]
    wheel = "\n".join(["Wheel-Version: 1.0" if ln.startswith("Wheel-Version:") else ln for ln in keep]
                      + ["Generator: epirefine dev/pure_wheel.py", "Root-Is-Purelib: true", "Tag: py3-none-any"]) + "\n"
    wheel_info = next(i for i in infos if i.filename == f"{dist_info}/WHEEL")
    record_info = next(i for i in infos if i.filename == f"{dist_info}/RECORD")
    files.append((wheel_info, wheel.encode()))
    rec = io.StringIO()
    w = csv.writer(rec, lineterminator="\n")
    for info, data in files:
        w.writerow([info.filename, _hash(data), len(data)])
    w.writerow([record_info.filename, "", ""])
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for info, data in files + [(record_info, rec.getvalue().encode())]:
            zi = zipfile.ZipInfo(info.filename, info.date_time)
            zi.external_attr, zi.compress_type = info.external_attr, zipfile.ZIP_DEFLATED
            z.writestr(zi, data)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wheel", type=Path, help="a compiled wheel of EpiRefine")
    ap.add_argument("--out", type=Path, default=None, help="directory for the pure wheel (default: the wheel's)")
    a = ap.parse_args()
    print(pure_wheel(a.wheel, a.out or a.wheel.parent))


if __name__ == "__main__":
    main()
