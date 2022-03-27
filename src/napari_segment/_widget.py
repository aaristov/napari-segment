"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from functools import partial

import dask
import napari
import numpy as np
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    label,
)
from skimage.measure import regionprops_table

# import matplotlib.pyplot as plt


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


def save_values(val):
    print(f"Saving {val}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
@magic_factory(
    auto_call=True,
    thr={
        "label": "Threshold",
        "widget_type": "FloatSlider",
        "min": 0.1,
        "max": 0.5,
    },
    erode={"widget_type": "Slider", "min": 1, "max": 10},
)
def segment_organoid(
    BF_layer: "napari.layers.Image",
    fluo_layer: "napari.layers.Image",
    thr: float = 0.4,
    erode: int = 10,
    donut=10,
) -> napari.types.LayerDataTuple:
    # frame = napari.current_viewer().cursor.position[0]
    kwargs = {}
    ddata = BF_layer.data
    labels = ddata.map_blocks(
        partial(segment_bf, thr=thr, erode=erode), dtype=ddata.dtype
    )
    selected_labels = dask.array.map_blocks(filter_biggest, labels, donut)
    print(selected_labels.shape)
    return [
        (labels, {"name": "raw_labels", "visible": False, **kwargs}, "labels"),
        (selected_labels, {"name": "selected labels", **kwargs}, "labels"),
    ]


def filter_biggest(labels, donut=10):
    props = regionprops_table(
        labels[0],
        properties=(
            "label",
            "area",
        ),
    )
    biggest_prop_index = np.argmax(props["area"])
    label_of_biggest_object = props["label"][biggest_prop_index]
    spheroid_mask = labels[0] == label_of_biggest_object
    bg_mask = np.bitwise_xor(
        binary_dilation(spheroid_mask, structure=np.ones((donut, donut))),
        spheroid_mask,
    )
    return (
        spheroid_mask.astype("uint16") + 2 * bg_mask.astype("uint16")
    ).reshape(labels.shape)


def segment_bf(well, thr=0.2, smooth=10, erode=10, fill=True, plot=False):
    """
    Serments input 2d array using thresholded gradient with filling
    Returns SegmentedImage object
    """
    grad = get_2d_gradient(well[0])
    sm = gaussian_filter(grad, smooth)
    #     sm = multiwell.gaussian_filter(well, smooth)

    regions = sm > thr * sm.max()

    if fill:
        regions = binary_fill_holes(regions)

    if erode:
        regions = binary_erosion(regions, iterations=erode)
    labels, _ = label(regions)

    return labels.reshape(well.shape)


def get_2d_gradient(xy):
    gx, gy = np.gradient(xy)
    return np.sqrt(gx**2 + gy**2)
