import os

import dask.array as da
import numpy as np

from napari_segment import SegmentStack


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_segment_stack(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.open_sample("napari-segment", "D3_D4")

    # create our widget, passing in the viewer
    widget = SegmentStack(viewer)
    assert widget.input.current_choice == "D3_D4"
    assert os.path.join(".napari-segment", "data", "D3_D4.nd2") in widget.path
    assert isinstance(widget.ddata, da.Array)

    widget.save_params()
    assert os.path.exists(widget.path + ".params.yaml")

    widget.plot_stats(True)

    widget.make_manual_layer()

    assert (mtitle := "Manual Labels") in viewer.layers
    assert isinstance(viewer.layers[mtitle].data, np.ndarray)

    # read captured output and check that it's as we expected
    # captured = capsys.readouterr()
    # assert captured.out == "napari has 1 layers\n"
