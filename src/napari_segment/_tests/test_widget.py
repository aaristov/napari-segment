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

    widget.save_csv()
    assert os.path.exists(widget.path + ".table.csv")
    assert os.path.exists(widget.path + ".labels.tif")


def test_manual_labels(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.open_sample("napari-segment", "D3_D4")
    _ = SegmentStack(viewer)
    assert (mtitle := "Manual Labels") in viewer.layers
    assert isinstance(viewer.layers[mtitle].data, np.ndarray)


def test_grad_stack(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.open_sample("napari-segment", "D3_D1")

    # create our widget, passing in the viewer
    widget = SegmentStack(viewer)
    assert widget.input.current_choice == "D3_D1"
    assert os.path.join(".napari-segment", "data", "D3_D1.nd2") in widget.path
    assert isinstance(widget.ddata, da.Array)

    widget.save_params()
    assert os.path.exists(widget.path + ".params.yaml")

    widget.plot_stats(True)

    widget.make_manual_layer()

    assert (mtitle := "Manual Labels") in viewer.layers
    assert isinstance(viewer.layers[mtitle].data, np.ndarray)

    if os.path.exists(csv := widget.path + ".table.csv"):
        os.remove(csv)
    if os.path.exists(tif := widget.path + ".labels.tif"):
        os.remove(tif)

    widget.save_csv()
    assert os.path.exists(widget.path + ".table.csv")
    assert os.path.exists(widget.path + ".labels.tif")
