import dask.array as da

from napari_segment._sample_data import make_late_aggregate


def test_late_agregate():
    for res in (make_late_aggregate(), make_late_aggregate()):
        assert isinstance(res, list)
        assert len(res[0]) == 3
        assert isinstance(res[0][0], da.Array)
