"""Shared fixtures: the synthetic community (tests/synth.py) and the runs several test files compare with
are made once per test session."""
import pytest

import synth
from small_community import write_community
from helpers import Run, copy_cache, logged, run_pipeline


@pytest.fixture(autouse=True)
def _kernels_state():
    """Each test starts with a fresh kernels state (epirefine._rs keeps the import once per process, and a run's
    decision until the run ends), so that tests which hide or patch the build see their own."""
    from epirefine import _rs
    _rs.reset()
    yield
    _rs.reset()


@pytest.fixture(scope="session")
def community(tmp_path_factory):
    """The default synthetic community (seed 1): input files plus the planted truth."""
    return synth.make_community(tmp_path_factory.mktemp("community"), seed=1)


@pytest.fixture(scope="session")
def base_run(community, tmp_path_factory) -> Run:
    """The reference run: epirefine.pipeline.run with default options, one worker, a cold cache."""
    return run_pipeline(**community.args(tmp_path_factory.mktemp("base") / "out"))


@pytest.fixture(scope="session")
def sisters(tmp_path_factory):
    """The synthetic community with sister genomes (synth.SISTERS, seed 1), where the switch of 0.11 picks relatives."""
    return synth.make_community(tmp_path_factory.mktemp("sisters"), seed=1, genomes=synth.SISTERS)


@pytest.fixture(scope="session")
def sisters_run(sisters, tmp_path_factory) -> Run:
    """The sister community at the default options (option E), one worker, a cold cache."""
    return run_pipeline(**sisters.args(tmp_path_factory.mktemp("sisters_e") / "out"))


@pytest.fixture(scope="session")
def sisters_switch_run(sisters, sisters_run, tmp_path_factory) -> Run:
    """The sister community with the deprecated --relatives switch (the path of 0.11.1), from sisters_run's cache."""
    out = copy_cache(sisters_run, tmp_path_factory.mktemp("sisters_switch") / "out")
    return run_pipeline(**sisters.args(out), cache_dir=str(out / "cache"), relatives="switch")


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


@pytest.fixture(scope="session")
def small_community(tmp_path_factory):
    """Paths of the three-genome community of tests/small_community.py (input, log and provenance tests)."""
    return write_community(tmp_path_factory.mktemp("small_community"))
