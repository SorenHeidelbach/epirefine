import numpy as np

from epirefine.refine import regional_pattern


def test_regional_pattern_separates_whole_foreign_member_and_mixed():
    rng = np.random.default_rng(0)
    llrs = []
    for _ in range(30):
        llrs.append(rng.normal(-8, 2, 10))                          # members: every chunk fits the bin
    for _ in range(5):
        llrs.append(rng.normal(12, 2, 10))                          # contaminants: every chunk foreign
    mixed = np.r_[rng.normal(-8, 2, 6), rng.normal(15, 2, 4)]      # chimeric: last 4 chunks foreign
    llrs.append(mixed)
    post, chunk_post, _ = regional_pattern(llrs)
    assert post[:30].argmax(1).tolist() == [0] * 30
    assert post[30:35].argmax(1).tolist() == [1] * 5
    assert post[35].argmax() == 2
    assert (chunk_post[35] >= 0.5).tolist() == [False] * 6 + [True] * 4
