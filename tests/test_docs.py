"""The docs stay in step with the code: every option of the command line is in docs/usage.md, and every file,
column and params.json key of a default run is named in docs/outputs.md (in backticks).

Outputs that only some options write (unbinned_prior.tsv with --unknown-prior unbinned, the temperature_by_sites
law with --temperature-law sites) are documented too, but are not run here; the P3 note in the dev log lists the
check that covered them.
"""
import argparse
import json
import re
from pathlib import Path

from epirefine.cli import build_parser

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _ticked(name: str) -> set[str]:
    return set(re.findall(r"`([^`\n]+)`", (DOCS / name).read_text()))


def _options(parser, path=("epirefine",)):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield from _options(sub, path + (name,))
        elif action.help != argparse.SUPPRESS:
            for opt in action.option_strings:
                if opt.startswith("--") and opt != "--help":
                    yield " ".join(path), opt


def test_every_command_line_option_is_in_usage():
    text = (DOCS / "usage.md").read_text()
    missing = [f"{cmd} {opt}" for cmd, opt in _options(build_parser())
               if not re.search(re.escape(opt) + r"(?![\w-])", text)]
    assert not missing, "options not in docs/usage.md: " + ", ".join(missing)


def _keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{prefix}{k}", k
            yield from _keys(v, f"{prefix}{k}.")


def test_every_output_file_column_and_params_key_is_in_outputs(base_run):
    named = _ticked("outputs.md")
    out = base_run.out
    missing = []
    for f in sorted(out.rglob("*")):
        rel = f.relative_to(out).as_posix()
        if f.is_dir() or rel.startswith("cache/"):
            continue
        if rel not in named and f.name not in named:
            missing.append(f"file {rel}")
        if f.suffix == ".tsv":
            missing += [f"{rel}: column {c}" for c in f.read_text().split("\n", 1)[0].split("\t") if c not in named]
    # a nested key counts as documented when its own name is (the docs name keys inside their parent's entry)
    missing += [f"params.json: {path}" for path, key in _keys(json.loads((out / "params.json").read_text()))
                if key not in named]
    assert not missing, "not documented in docs/outputs.md:\n  " + "\n  ".join(missing)
