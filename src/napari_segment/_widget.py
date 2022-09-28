"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
import os
from functools import partial

import dask
import magicgui.widgets as w
import napari
import numpy as np
import qtpy.QtWidgets as q
import yaml
from magicgui import magic_factory
from napari.layers import Image
from napari.utils.notifications import show_error, show_info
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    label,
)
from skimage.measure import regionprops

# import matplotlib.pyplot as plt


class ExampleQWidget(q.QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.input = w.ComboBox(
            label="BF data",
            annotation=Image,
            choices=[
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
            ],
        )
        self.input.changed.connect(self.restore_params)
        self.input.changed.connect(self.preprocess)

        self.binning_widget = w.RadioButtons(
            label="binning",
            choices=[2**n for n in range(4)],
            value=4,
            orientation="horizontal",
        )
        self.binning_widget.changed.connect(self.preprocess)

        self.thr = w.FloatSlider(label="Threshold", min=0.1, max=0.9)
        self.thr.changed.connect(self.threshold)

        self.erode = w.SpinBox(label="erode", min=0, max=10)
        self.erode.changed.connect(self.threshold)

        self.use = w.RadioButtons(
            label="Use",
            choices=["Intensity", "Gradient", "GDif"],
            value="Intensity",
            orientation="horizontal",
        )
        self.use.changed.connect(self.preprocess)

        self.smooth = w.SpinBox(label="smooth", min=0, max=10)
        self.smooth.changed.connect(self.preprocess)

        self.min_diam = w.Slider(
            label="Min_diameter",
            min=1,
            max=500,
        )
        self.min_diam.changed.connect(self.update_out)

        self.max_diam = w.Slider(
            label="Max_diameter", min=150, max=2000, step=150
        )
        self.max_diam.changed.connect(self.update_out)

        self.max_ecc = w.FloatSlider(
            label="Max eccentricity",
            min=0.0,
            max=1.0,
        )
        self.max_ecc.changed.connect(self.update_out)

        self.btn = q.QPushButton("Save!")
        self.btn.clicked.connect(self.save_params)

        self.container = w.Container(
            widgets=[
                self.input,
                w.Label(label="Prepocessing"),
                self.binning_widget,
                self.use,
                w.Label(label="Detection"),
                self.smooth,
                self.thr,
                self.erode,
                w.Label(label="Filters"),
                self.min_diam,
                self.max_diam,
                self.max_ecc,
            ]
        )

        self.setLayout(q.QVBoxLayout())
        self.layout().addWidget(self.container.native)
        self.layout().addWidget(self.btn)
        self.layout().addStretch()

        self.viewer.layers.events.inserted.connect(self.reset_choices)
        self.viewer.layers.events.removed.connect(self.reset_choices)

        if self.input.current_choice:
            print("start")
            self.restore_params()
            self.preprocess()

    def preprocess(self):
        self.binning = self.binning_widget.value
        try:
            self.data = self.viewer.layers[self.input.current_choice].data[
                ..., :: self.binning, :: self.binning
            ]
        except KeyError:
            show_error("No data to process")
            return
        self.scale = np.ones((len(self.data.shape),))
        self.scale[-2:] = self.binning
        if isinstance(self.data, np.ndarray):
            chunksize = np.ones(len(self.data.shape))
            chunksize[-2:] = self.data.shape[-2:]  # xy full size
            self.ddata = dask.array.from_array(
                self.data.astype("f"), chunks=chunksize
            )
        else:
            self.ddata = self.data.astype("f")

        show_info(self.use.value)
        if self.use.value == "Gradient":
            self.smooth_gradient = self.ddata.map_blocks(
                partial(get_gradient, smooth=self.smooth.value),
                dtype=self.ddata.dtype,
            )
        elif self.use.value == "Intensity":
            self.smooth_gradient = self.ddata.map_blocks(
                lambda d: 1 - norm01(gaussian_filter(d, self.smooth.value)),
                dtype=self.ddata.dtype,
            )
        elif self.use.value == "GDif":
            self.smooth_gradient = self.ddata.map_blocks(
                lambda d: (
                    gaussian_filter(d, self.smooth.value)
                    - gaussian_filter(d, self.smooth.value + 2)
                ),
                dtype=self.ddata.dtype,
            )
        else:
            self.smooth_gradient = np.zeros_like(self.ddata)
            raise (
                ValueError(
                    f"""Filter `{self.use.value}` not understood!
                    Expected `Gradient` or `Intensity`  or `GDif`"""
                )
            )

        if not (name := "Preprocessing") in self.viewer.layers:
            self.viewer.add_image(
                data=self.smooth_gradient,
                **{"name": name, "scale": self.scale},
            )
        else:
            self.viewer.layers[name].data = self.smooth_gradient
            self.viewer.layers[name].scale = self.scale
        self.threshold()

    def threshold(self):
        if not self.input.current_choice:
            return
        self.labels = self.smooth_gradient.map_blocks(
            partial(
                threshold_gradient, thr=self.thr.value, erode=self.erode.value
            ),
            dtype=np.int32,
        )
        if not (name := "Detections") in self.viewer.layers:
            self.viewer.add_labels(
                data=self.labels,
                opacity=0.3,
                **{"name": name, "scale": self.scale},
            )
        else:
            self.viewer.layers[name].data = self.labels
            self.viewer.layers[name].scale = self.scale
            self.viewer.layers[name].contour = 5
        self.update_out()

    def update_out(self):

        if not self.input.current_choice:
            return

        try:
            selected_labels = dask.array.map_blocks(
                filter_labels,
                self.labels,
                self.min_diam.value / self.binning,
                self.max_diam.value / self.binning,
                self.max_ecc.value,
                dtype=np.int32,
            )

            if not (name := "selected labels") in self.viewer.layers:
                self.viewer.add_labels(
                    data=selected_labels,
                    opacity=0.5,
                    **{"name": name, "scale": self.scale},
                )
            else:
                self.viewer.layers[name].scale = self.scale
                self.viewer.layers[name].data = selected_labels
            # self.save_params()
        except TypeError as e:
            show_error(f"Relax filter! {e}")

    def save_params(self):
        data = {
            "binning": self.binning,
            "use": self.use.value,
            "smooth": self.smooth.value,
            "thr": self.thr.value,
            "erode": self.erode.value,
            "min_diameter": self.min_diam.value,
            "max_diameter": self.max_diam.value,
            "max_ecc": self.max_ecc.value,
        }
        try:
            path = self.viewer.layers[self.input.current_choice].metadata[
                "path"
            ]
            dir = os.path.dirname(path)
            filename = os.path.basename(path)
            new_name = filename + ".params.yaml"

            with open(os.path.join(dir, new_name), "w") as f:
                yaml.safe_dump(data, f)
            show_info(f"Parameters saves into {new_name}")
        except KeyError:
            show_error("Saving parameters failed")
        with open((ff := ".latest.params.yaml"), "w") as f:
            yaml.safe_dump(data, f)
            show_info(f"Parameters saves into {ff}")

    def restore_params(self):
        try:
            path = self.viewer.layers[self.input.current_choice].metadata[
                "path"
            ]
            dir = os.path.dirname(path)
            filename = os.path.basename(path)
            new_name = filename + ".params.yaml"
        except KeyError:
            pass
        try:
            with open(ppp := os.path.join(dir, new_name)) as f:
                data = yaml.safe_load(f)
            show_info(f"restoring parameters from {new_name}")

        except (UnboundLocalError, UnicodeDecodeError):
            with open(ppp := ".latest.params.yaml") as f:
                data = yaml.safe_load(f)
            show_info(f"restoring parameters from {ppp}")
        print(data)
        try:
            self.binning_widget.value = data["binning"]
            self.use.value = data["use"]
            self.smooth.value = data["smooth"]
            self.thr.value = data["thr"]
            self.erode.value = data["erode"]
            self.min_diam.value = data["min_diameter"]
            self.max_diam.value = data["max_diameter"]
            self.max_ecc.value = data["max_ecc"]
        except Exception as e:
            show_error(f"Restore settings failed, {e}")

    def reset_choices(self, event=None):
        self.input.reset_choices(event)
        self.input.choices = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image) and layer.name != "Preprocessing"
        ]
        # self.restore_params()


def norm01(data):
    d = data.copy()
    return (a := d - d.min()) / a.max()


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
    bin={
        "label": "binning",
        "widget_type": "ComboBox",
        "choices": [1, 2, 4, 8, 16],
    },
    thr={
        "label": "Threshold",
        "widget_type": "FloatSlider",
        "min": 0.1,
        "max": 0.9,
    },
    erode={"widget_type": "Slider", "min": 1, "max": 10},
    min_diam={"widget_type": "Slider", "min": 100, "max": 1000},
    max_diam={"widget_type": "Slider", "min": 150, "max": 2000},
    max_ecc={
        "label": "Eccentricity",
        "widget_type": "FloatSlider",
        "min": 0.0,
        "max": 1.0,
    },
)
def segment_organoid(
    BF_layer: "napari.layers.Image",
    bin: int = 1,
    use_gradient=True,
    smooth=10,
    thr: float = 0.3,
    erode: int = 10,
    min_diam=150,
    max_diam=550,
    max_ecc=0.7,
    show_detections=True,
) -> napari.types.LayerDataTuple:
    # frame = napari.current_viewer().cursor.position[0]

    ddata = BF_layer.data[..., ::bin, ::bin]
    if isinstance(ddata, np.ndarray):
        chunksize = np.ones(ddata.ndims)
        chunksize[-2:] = ddata.shape[-2:]  # xy full size
        ddata = dask.array.from_array(ddata, chunksize=chunksize)
    smooth_gradient = (
        ddata.map_blocks(
            partial(get_gradient, smooth=smooth), dtype=ddata.dtype
        )
        if use_gradient
        else ddata.map_blocks(
            lambda d: 1
            - (a := (s := gaussian_filter(d, smooth)) - s.min()) / a.max(),
            dtype=ddata.dtype,
        )
    )
    labels = smooth_gradient.map_blocks(
        partial(threshold_gradient, thr=thr, erode=erode),
        dtype=ddata.dtype,
    )
    try:
        selected_labels = dask.array.map_blocks(
            filter_labels,
            labels,
            min_diam / bin,
            max_diam / bin,
            max_ecc,
            dtype=ddata.dtype,
        )
        scale = np.ones((len(labels.shape),))
        scale[-2:] = bin
        kwargs = {"scale": scale}
        return [
            (
                smooth_gradient,
                {"name": "Gradient", "visible": show_detections, **kwargs},
                "image",
            ),
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
                {
                    "name": "Detections",
                    "visible": True,
                    "scale": scale,
                    **kwargs,
                },
                "labels",
            ),
        ]


def filter_labels(labels, min_diam=50, max_diam=150, max_ecc=0.2):
    if max_diam <= min_diam:
        min_diam = max_diam - 10
    data = strip_dimensions(labels)
    props = regionprops(
        data,
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
    mask = np.sum([data == v for v in good_labels], axis=0)
    # print(mask.shape)
    return (label(mask)[0].astype("uint16")).reshape(labels.shape)


def get_gradient(bf_data: np.ndarray, smooth=10, bin=1):
    """
    Removes first dimension,
    Computes gradient of the image,
    applies gaussian filter
    Returns SegmentedImage object
    """
    data = strip_dimensions(bf_data)
    gradient = get_2d_gradient(data[::bin, ::bin])
    smoothed_gradient = gaussian_filter(gradient, smooth)
    #     sm = multiwell.gaussian_filter(well, smooth)
    return norm01(smoothed_gradient.reshape(bf_data[..., ::bin, ::bin].shape))


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


def strip_dimensions(array: np.ndarray):
    data = array.copy()
    while data.ndim > 2:
        assert (
            data.shape[0] == 1
        ), f"Unexpected multidimensional data! {data.shape}"
        data = data[0]
    return data


def get_2d_gradient(xy):
    gx, gy = np.gradient(xy)
    return np.sqrt(gx**2 + gy**2)
