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
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    label,
)
from skimage.measure import regionprops

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
    min_diam={"widget_type": "Slider", "min": 10, "max": 250},
    max_diam={"widget_type": "Slider", "min": 150, "max": 1000},
    max_ecc={
        "label": "Eccentricity",
        "widget_type": "FloatSlider",
        "min": 0.0,
        "max": 1.0,
    },
)
def segment_organoid(
    BF_layer: "napari.layers.Image",
    thr: float = 0.3,
    erode: int = 10,
    min_diam=150,
    max_diam=550,
    max_ecc=0.7,
    show_detections=True,
) -> napari.types.LayerDataTuple:
    # frame = napari.current_viewer().cursor.position[0]
    kwargs = {"scale": BF_layer.scale}
    print(kwargs)
    ddata = BF_layer.data
    if isinstance(ddata, np.ndarray):
        chunksize = np.ones(ddata.ndims)
        chunksize[-2:] = ddata.shape[-2:]  # xy full size
        ddata = dask.array.from_array(ddata, chunksize=chunksize)
    smooth_gradient = ddata.map_blocks(
        partial(get_gradient), dtype=ddata.dtype
    )
    labels = smooth_gradient.map_blocks(
        partial(threshold_gradient, thr=thr, erode=erode),
        dtype=ddata.dtype,
    )
    try:
        selected_labels = dask.array.map_blocks(
            filter_labels,
            labels,
            min_diam,
            max_diam,
            max_ecc,
            dtype=ddata.dtype,
        )
        # print(selected_labels.shape)
        return [
            (
                labels,
                {"name": "Detections", "visible": show_detections, **kwargs},
                "labels",
            ),
            (selected_labels, {"name": "selected labels", **kwargs}, "labels"),
        ]
    except TypeError:
        return [
            (
                labels,
                {"name": "Detections", "visible": True, **kwargs},
                "labels",
            ),
        ]


def filter_labels(labels, min_diam=50, max_diam=150, max_ecc=0.2):
    if max_diam <= min_diam:
        raise ValueError(
            "min value is greater than max value for the diameter filter"
        )
    props = regionprops(
        labels[0],
    )
    good_props = filter(
        lambda p: (d := p.major_axis_length) > min_diam
        and d < max_diam
        and p.eccentricity < max_ecc,
        props,
    )
    good_labels = [p.label for p in good_props]
    if len(good_labels) < 1:
        return np.zeros_like(labels)
    # print(f'good_labels {good_labels}')
    mask = np.sum([labels == v for v in good_labels], axis=0)
    # print(mask.shape)
    return (label(mask)[0].astype("uint16")).reshape(labels.shape)


def get_gradient(bf_data: np.ndarray, smooth=10):
    """
    Removes first dimension,
    Computes gradient of the image,
    applies gaussian filter
    Returns SegmentedImage object
    """
    data = strip_dimensions(bf_data)
    gradient = get_2d_gradient(data)
    smoothed_gradient = gaussian_filter(gradient, smooth)
    #     sm = multiwell.gaussian_filter(well, smooth)
    return smoothed_gradient.reshape(bf_data.shape)


def threshold_gradient(
    smoothed_gradient: np.ndarray,
    thr: float = 0.4,
    fill: bool = True,
    erode: int = 1,
):
    data = strip_dimensions(smoothed_gradient)
    regions = data > thr * data.max()

    if fill:
        regions = binary_fill_holes(regions)

    if erode and erode > 0:
        regions = binary_erosion(regions, iterations=erode)
    labels, _ = label(regions)

    return labels.reshape(smoothed_gradient.shape)


def strip_dimensions(array:np.ndarray):
    data = array.copy()
    while data.ndim > 2:
        assert data.shape[0] == 1, f'Unexpected multidimensional data! {data.shape}'
        data = data[0]
    return data

def get_2d_gradient(xy):
    gx, gy = np.gradient(xy)
    return np.sqrt(gx**2 + gy**2)
