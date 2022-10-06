from napari_segment import SegmentStack
import numpy as np
import dask.array as da

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_segment_stack(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((5, 100, 100)), name="test_data", metadata={"path": "fakepath.nd2"})

    # create our widget, passing in the viewer
    widget = SegmentStack(viewer)
    assert widget.input.current_choice == "test_data"
    assert widget.path == "fakepath.nd2"
    assert isinstance(widget.ddata, da.Array) 
    assert widget.ddata.chunks == ((1, 1, 1, 1, 1), (100/widget.binning,), (100/widget.binning,))

    
    # read captured output and check that it's as we expected
    # captured = capsys.readouterr()
    # assert captured.out == "napari has 1 layers\n"
