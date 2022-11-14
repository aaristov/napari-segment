import dask.array as da

from napari_segment._sample_data import make_late_aggregate


def test_early_aggregate():
    res = make_late_aggregate()
    assert isinstance(res, tuple)
    assert isinstance(res[0], da.Array)


def test_late_agregate():
    res = make_late_aggregate()
    assert isinstance(res, tuple)
    assert isinstance(res[0], da.Array)
