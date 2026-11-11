"""Shared fixtures: the synthetic community (tests/synth.py) and the runs several test files compare with
are made once per test session."""
import pytest

import synth
from helpers import Run, logged, run_pipeline


@pytest.fixture(scope="session")
def community(tmp_path_factory):
    """The default synthetic community (seed 1): input files plus the planted truth."""
    return synth.make_community(tmp_path_factory.mktemp("community"), seed=1)


@pytest.fixture(scope="session")
def base_run(community, tmp_path_factory) -> Run:
    """The reference run: epirefine.pipeline.run with default options, one worker, a cold cache."""
    return run_pipeline(**community.args(tmp_path_factory.mktemp("base") / "out"))


@pytest.fixture(scope="session")
def many_ranges():
    """Make the pileup pass split even a small pileup into byte ranges (the pipeline splits only files
    above 64 MB), so that runs with several workers really read the pileup in several pieces."""
    from epirefine import sites
    real = sites.split_pileup
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sites, "split_pileup", lambda path, n_parts, min_size=0: real(path, n_parts, min_size=0))
        yield


@pytest.fixture(scope="session")
def cli_run(community, tmp_path_factory, many_ranges) -> Run:
    """The same community through the command line, with three workers on a pileup read in several ranges."""
    from epirefine.cli import main
    out = tmp_path_factory.mktemp("cli") / "out"
    return Run(out, logged(main, community.cli_args(out, "--workers", 3, "--motif-fit")))
