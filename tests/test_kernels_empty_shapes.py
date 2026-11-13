"""The model kernels on tables with a zero-size axis (no groups, no grid points, no bins, no columns): the numpy
results, or a ValueError for a grid of no points, never a Rust panic (pyo3's PanicException is a BaseException that
`except Exception` does not catch). B6 review, should-fix "guard zero-size shapes"."""
import numpy as np
import pytest

k = pytest.importorskip("epirefine._kernels", reason="no compiled kernels in this install")
f32 = np.float32


def test_bin_loglik_without_groups_or_grid():
    for shape in ((5, 0, 64), (5, 3, 0)):
        out = k.bin_loglik(np.zeros(shape, f32), np.zeros(5, np.int64), 2, np.ones(5))
        assert out.shape == (2,) + shape[1:] and not out.any()


def test_terms_without_groups_are_zero():
    """A sum over no groups is 0 for every unit (numpy: an empty sum)."""
    assert np.array_equal(k.community_terms(np.zeros((5, 0, 64), f32), np.zeros((0, 64))), np.zeros(5))
    out = k.leave_chunk_out_terms(np.zeros((5, 0, 64), f32), np.zeros(5, np.int64), np.ones(5), np.zeros((2, 0, 64)),
                                  np.zeros((0, 64)))
    assert np.array_equal(out, np.zeros(5))


def test_empty_grid_is_a_value_error():
    with pytest.raises(ValueError, match="grid has no points"):
        k.community_terms(np.zeros((5, 3, 0), f32), np.zeros((3, 0)))
    with pytest.raises(ValueError, match="grid has no points"):
        k.leave_chunk_out_terms(np.zeros((5, 3, 0), f32), np.zeros(5, np.int64), np.ones(5), np.zeros((2, 3, 0)),
                                np.zeros((3, 0)))


def test_cell_sums_without_columns():
    out = k.cell_sums(np.zeros((4, 0)), np.array([0, 1, 2], np.int64), np.ones(3), np.array([0, 2], np.int64))
    assert out.shape == (2, 0)


def test_score_sums_without_bins():
    s, r = k.score_sums(np.zeros((5, 3, 4)), np.zeros((5, 3)), np.zeros((3, 4, 0)), np.zeros((3, 0)),
                        le0=np.zeros((5, 3)))
    assert s.shape == (5, 0) and r.shape == (5, 0)
