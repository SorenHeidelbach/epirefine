"""dev/pure_wheel.py derives the py3-none-any wheel from a compiled one: it drops the extension and refuses to ship
any other compiled file under a pure tag (B6 review, should-fix). Skipped where dev/ is not shipped (an sdist)."""
import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dev" / "pure_wheel.py"
pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="no dev/pure_wheel.py here")
DI = "epirefine-1.0.dist-info"
WHEEL = "Wheel-Version: 1.0\nGenerator: maturin\nRoot-Is-Purelib: false\nTag: cp310-abi3-manylinux_2_17_x86_64\n"


def _module():
    spec = importlib.util.spec_from_file_location("pure_wheel", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _compiled(path: Path, extra: dict | None = None, wheel: str = WHEEL) -> Path:
    files = {"epirefine/__init__.py": "x = 1\n", "epirefine/_kernels/__init__.py": "",
             "epirefine/_kernels/_core.abi3.so": b"\x7fELF...", f"{DI}/sboms/x.json": "{}",
             f"{DI}/METADATA": "Name: epirefine\nVersion: 1.0\n", f"{DI}/WHEEL": wheel, f"{DI}/RECORD": "",
             **(extra or {})}
    with zipfile.ZipFile(path, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return path


def test_pure_wheel_drops_the_extension(tmp_path):
    pw = _module()
    out = pw.pure_wheel(_compiled(tmp_path / "epirefine-1.0-cp310-abi3-manylinux_2_17_x86_64.whl"), tmp_path / "o")
    assert out.name == "epirefine-1.0-py3-none-any.whl"
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert "epirefine/_kernels/_core.abi3.so" not in names and not any("sboms" in n for n in names)
        assert "Tag: py3-none-any" in z.read(f"{DI}/WHEEL").decode()


@pytest.mark.parametrize("extra", [{"epirefine.libs/libdeflate-1a2b.so.0": b"\x7fELF"},
                                   {"epirefine/.dylibs/libz.dylib": b"\xcf\xfa\xed\xfe"},
                                   {"epirefine/data/blob.bin": b"\x7fELF hidden"}])
def test_pure_wheel_refuses_other_binaries(tmp_path, extra):
    pw = _module()
    with pytest.raises(SystemExit, match="compiled files"):
        pw.pure_wheel(_compiled(tmp_path / "epirefine-1.0-cp310-abi3-manylinux_2_17_x86_64.whl", extra), tmp_path / "o")


def test_pure_wheel_of_a_wheel_with_two_platform_tags(tmp_path):
    """maturin's manylinux2014 wheels name the build twice (manylinux2014 and manylinux_2_17): that is one build."""
    pw = _module()
    wheel = WHEEL + "Tag: cp310-abi3-manylinux2014_x86_64\n"
    src = _compiled(tmp_path / "epirefine-1.0-cp310-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", wheel=wheel)
    with zipfile.ZipFile(pw.pure_wheel(src, tmp_path / "o")) as z:
        assert [ln for ln in z.read(f"{DI}/WHEEL").decode().splitlines() if ln.startswith("Tag:")] == ["Tag: py3-none-any"]


def test_pure_wheel_wants_abi3_tags(tmp_path):
    pw = _module()
    wheel = WHEEL.replace("cp310-abi3", "cp312-cp312")
    with pytest.raises(SystemExit, match="abi3"):
        pw.pure_wheel(_compiled(tmp_path / "epirefine-1.0-cp312-cp312-linux_x86_64.whl", wheel=wheel), tmp_path / "o")
