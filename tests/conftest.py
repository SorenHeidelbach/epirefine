"""Shared fixtures: the synthetic community (tests/synth.py) is written once per test session."""
import pytest

import synth


@pytest.fixture(scope="session")
def community(tmp_path_factory):
    """The default synthetic community (seed 1): input files plus the planted truth."""
    return synth.make_community(tmp_path_factory.mktemp("community"), seed=1)
