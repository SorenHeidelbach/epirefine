"""rust/THIRD_PARTY_LICENSES.txt (shipped in the wheels and the sdist) lists the crates of the current rust/Cargo.lock."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("cargo") is None or not (ROOT / "rust").is_dir(),
                    reason="needs cargo and the source tree")
def test_third_party_licences_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "dev" / "bundle_licenses.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
