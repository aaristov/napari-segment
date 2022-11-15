def test_late_agregate(make_napari_viewer):

    viewer = make_napari_viewer()
    viewer.open_sample("napari-segment", "D3_D4")
    viewer.open_sample("napari-segment", "D3_D1")
    assert "D3_D4" in viewer.layers
    assert "D3_D1" in viewer.layers
