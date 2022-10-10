"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
import logging
import os
from enum import Enum
from functools import partial, reduce
from importlib.metadata import PackageNotFoundError, version

import dask
import magicgui.widgets as w
import napari
import numpy as np
import pandas as pd
import qtpy.QtWidgets as q
import yaml
from magicgui import magic_factory
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from napari.layers import Image, Labels
from napari.utils.notifications import show_error, show_info
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    label,
)
from skimage.measure import regionprops
from skimage.segmentation import clear_border

try:
    __version__ = version("napari-segment")
except PackageNotFoundError:
    # package is not installed
    __version__ = "Unknown"

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s : %(message)s"
)
logger = logging.getLogger("napari_segment._widget")

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s : %(message)s"
)
ff = logging.FileHandler("napari-segment.log")
ff.setFormatter(formatter)
logger.addHandler(ff)


class SegmentStack(q.QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter

    class Choices(Enum):
        INT = "Invert"
        GRAD = "Gradient"
        GDIF = "Gauss diff"

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.input = w.ComboBox(
            label="BF data",
            annotation=Image,
            choices=self.update_images(),
        )

        self.stat_layer_selector = w.ComboBox(
            label="Labels to quantify",
            annotation=Labels,
            choices=self.update_labels(),
        )

        self.btn_make_manual_labels = w.PushButton(
            text="Clone for manual correction"
        )
        self.btn_make_manual_labels.clicked.connect(self.make_manual_layer)

        self.stat_layer_selector_container = w.Container(
            widgets=[self.stat_layer_selector, self.btn_make_manual_labels],
            layout="horizontal",
        )

        self.binning_widget = w.RadioButtons(
            label="binning",
            choices=[2**n for n in range(4)],
            value=4,
            orientation="horizontal",
        )

        self.thr = w.FloatSlider(
            label="Threshold", min=0.1, max=0.9, value=0.4
        )

        self.erode = w.SpinBox(label="erode", min=0, max=10, value=0)

        self.use = w.RadioButtons(
            label="Use",
            choices=[v.value for v in self.Choices],
            value=self.Choices.INT.value,
            orientation="horizontal",
            allow_multiple=True,
        )

        self.smooth = w.SpinBox(label="smooth", min=0, max=10, value=2)

        self.min_diam = w.Slider(
            label="Min_diameter",
            min=1,
            max=500,
        )

        self.max_diam = w.Slider(
            label="Max_diameter", min=150, max=2000, step=150
        )

        self.max_ecc = w.FloatSlider(
            label="Max eccentricity", min=0.0, max=1.0, value=0.9
        )

        self.btn_save_params = q.QPushButton("Save params!")
        self.btn_save_csv = q.QPushButton("Save csv!")

        self.check_auto_plot = w.CheckBox(label="Auto Update")
        self.check_auto_plot.changed.connect(
            partial(self.plot_stats, force=True)
        )

        self.btn_update_stats = w.PushButton(text="Update plots")
        self.btn_update_stats.clicked.connect(self.plot_stats)

        self.update_plot_container = w.Container(
            widgets=[self.btn_update_stats, self.check_auto_plot],
            layout="horizontal",
        )

        self.pixel_size = 1
        self.pixel_unit = "px"
        self.pixel_size_widget = w.LineEdit(
            label="Pixel size",
            value=self.pixel_size,
        )  # bind=self.set_pixel_size)
        self.pixel_unit_widget = w.LineEdit(
            label="unit", value=self.pixel_unit
        )  # , bind=self.set_pixel_unit)
        self.pixel_block = w.Container(
            widgets=[self.pixel_size_widget, self.pixel_unit_widget],
            layout="horizontal",
        )

        self.canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.ax = self.canvas.figure.subplots(nrows=3, sharex=False)
        self.ax[0].set_title("Number of detections")
        self.ax[1].set_title(diams_title := "Diameters")
        self.diams_title = diams_title
        self.ax[2].set_title("Eccentricities")
        self.canvas.figure.tight_layout()

        (self._count,) = self.ax[0].plot(
            range(10), [0] * 10, "o", picker=True, pickradius=5
        )
        (self._diams,) = self.ax[1].plot(
            range(10), [0] * 10, "o", picker=True, pickradius=5
        )
        (self._eccs,) = self.ax[2].plot(
            range(10), [0] * 10, "o", picker=True, pickradius=5
        )

        self.container = w.Container(
            label="Container",
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
            ],
        )

        self.setLayout(q.QVBoxLayout())
        self.layout().addWidget(self.container.native)
        self.layout().addWidget(self.btn_save_params)
        self.layout().addWidget(self.update_plot_container.native)
        self.layout().addWidget(self.stat_layer_selector_container.native)
        self.layout().addWidget(self.pixel_block.native)
        self.layout().addWidget(self.canvas)
        self.layout().addWidget(NavigationToolbar(self.canvas, self))
        self.layout().addWidget(self.btn_save_csv)

        self.layout().addStretch()

        self.viewer.layers.events.inserted.connect(self.reset_choices)
        self.viewer.layers.events.removed.connect(self.reset_choices)
        self.input.changed.connect(self.restore_params)
        self.input.changed.connect(self.preprocess)

        self.canvas.mpl_connect("button_press_event", self.onfigclick)
        # self.canvas.mpl_connect("pick_event", self.on_pick)

        logger.debug("Initialization finished.")

        if self.input.current_choice:
            self.restore_params()
            self.preprocess()

    def update_labels(self):
        return [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Labels)
        ]

    def update_images(self):
        return [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
        ]

    def make_manual_layer(self):
        try:
            clone = self.selected_labels.compute()
            self.viewer.add_labels(
                data=clone,
                name="Manual Labels",
                scale=self.scale,
                metadata={
                    "binning": self.binning,
                    "source": self.viewer.layers["selected labels"],
                },
            )
            self.stat_layer_selector.choices = self.update_labels()[::-1]

            self.plot_stats(force=True)

        except Exception as e:
            show_error(err := f"Unable to create manual layer: {e}")
            logger.error(err)

    def move_step(self, slice):
        cur_ = self.viewer.dims.current_step
        new_ = list(cur_)
        new_[0] = slice
        self.viewer.dims.current_step = tuple(new_)

    def onfigclick(self, event):
        print(
            "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
            % (
                "double" if event.dblclick else "single",
                event.button,
                event.x,
                event.y,
                event.xdata,
                event.ydata,
            )
        )
        self.move_step(np.round(event.xdata, 0).astype(int))

    def on_pick(self, event):
        line = event.artist
        xdata, ydata = line.get_data()
        ind = event.ind
        print(f"on pick line: {np.array([xdata[ind], ydata[ind]]).T}")

        self.move_step(xdata[ind])

    def _invert(self, data2D):
        return 1 - norm01(gaussian_filter(data2D, self.smooth.value))

    def _grad(self, data2D):
        return get_gradient(data2D, smooth=self.smooth.value)

    def _gdif(self, data2D):
        return gaussian_filter(data2D, self.smooth.value) - gaussian_filter(
            data2D, self.smooth.value + 2
        )

    def set_pixel_size(self, value):
        self.pixel_size = value
        self.pixel_size_widget.value = value

    def set_pixel_unit(self, unit):
        self.pixel_unit = unit
        self.pixel_unit_widget.value = unit

    def preprocess(self):
        if not self.input.current_choice:
            return
        logger.debug(f"start preprocessing {self.input.current_choice}")

        try:
            self.path = self.viewer.layers[self.input.current_choice].metadata[
                "path"
            ]
        except KeyError:
            self.path = ""

        self.binning = self.binning_widget.value
        try:
            self.data = self.viewer.layers[self.input.current_choice].data[
                ..., :: self.binning, :: self.binning
            ]
            logger.debug(f"data after binning: {self.data}")

        except KeyError:
            return

        try:
            pixel_size = self.viewer.layers[
                self.input.current_choice
            ].metadata["pixel_size_um"]
            unit = "um"
            self.set_pixel_size(pixel_size)
            self.set_pixel_unit(unit)

        except KeyError:
            self.pixel_size = 1
            self.pixel_unit = "px"

        logger.debug(f"pixel size: {self.pixel_size} {self.pixel_unit}")

        self.scale = np.ones((len(self.data.shape),))
        self.scale[-2:] = self.binning
        logger.debug(f"Computed scale for napari {self.scale}")

        if isinstance(self.data, np.ndarray):
            chunksize = np.ones(len(self.data.shape))
            chunksize[-2:] = self.data.shape[-2:]  # xy full size
            self.ddata = dask.array.from_array(
                self.data.astype("f"), chunks=chunksize
            )
        else:
            self.ddata = self.data.astype("f")

        logger.debug(f"Dask array: {self.ddata}")

        logger.debug(f"Processing data with {self.use.value}")
        # show_info(self.use.value)
        if self.use.value == self.Choices.GRAD.value:
            self.smooth_gradient = self.ddata.map_blocks(
                self._grad,
                dtype=self.ddata.dtype,
            )
        elif self.use.value == self.Choices.INT.value:
            self.smooth_gradient = self.ddata.map_blocks(
                self._invert,
                dtype=self.ddata.dtype,
            )
        elif self.use.value == self.Choices.GDIF.value:
            self.smooth_gradient = self.ddata.map_blocks(
                self._gdif,
                dtype=self.ddata.dtype,
            )
        else:
            self.smooth_gradient = np.zeros_like(self.ddata)
            logger.error(
                f"""Filter `{self.use.value}` not understood!
                    Expected {[v.value for v in self.Choices]}"""
            )
            raise (
                ValueError(
                    f"""Filter `{self.use.value}` not understood!
                    Expected {[v.value for v in self.Choices]}"""
                )
            )

        if not (name := "Preprocessing") in self.viewer.layers:
            logger.debug("No Preprocessing layer found, adding one")
            self.layer_with_preprocessing = self.viewer.add_image(
                data=self.smooth_gradient,
                **{"name": name, "scale": self.scale},
            )
        else:
            logger.debug(
                f"Updating Preprocessing layer with \
                {self.smooth_gradient}"
            )
            self.layer_with_preprocessing.data = self.smooth_gradient
            self.layer_with_preprocessing.scale = self.scale

        logger.debug(
            f"Preprocessing finished, \
            sending {self.smooth_gradient} to thresholding"
        )
        self.threshold()

    def threshold(self):
        logger.debug("Start thresholding step")
        if not self.input.current_choice:
            return
        logger.debug(
            f"Thresholding with the thr={self.thr.value} \
            and erode={self.erode.value}"
        )
        self.labels = self.smooth_gradient.map_blocks(
            partial(
                threshold_gradient, thr=self.thr.value, erode=self.erode.value
            ),
            dtype=np.int32,
        )
        if not (name := "Detections") in self.viewer.layers:
            logger.debug("No Detections layer found, adding one")
            self.layer_with_detections = self.viewer.add_labels(
                data=self.labels,
                opacity=0.3,
                **{"name": name, "scale": self.scale},
            )

        else:
            logger.debug(f"Updating Detections layer with {self.labels}")
            self.layer_with_detections.data = self.labels
            self.layer_with_detections.scale = self.scale

        self.layer_with_detections.contour = 8 // self.binning

        logger.debug(
            f"Thresholding succesful. \
            Sending labels {self.labels} to filtering"
        )
        self.update_out()

    def update_out(self):

        if not self.input.current_choice:
            return
        logger.debug("Start filtering")
        try:
            logger.debug(
                f"""Filtering labels by  \
                ({self.min_diam.value / self.binning} \
             < size[px] < {self.max_diam.value / self.binning}) \
             and eccentricity > {self.max_ecc.value}"""
            )

            selected_labels = dask.array.map_blocks(
                filter_labels,
                self.labels,
                self.min_diam.value / self.binning,
                self.max_diam.value / self.binning,
                self.max_ecc.value,
                dtype=np.int32,
            )
            self.selected_labels = selected_labels

            if not (name := "selected labels") in self.viewer.layers:
                logger.debug("No selected labels layer found, adding one")

                self.layer_with_selected_labels = self.viewer.add_labels(
                    data=selected_labels,
                    opacity=0.5,
                    **{"name": name, "scale": self.scale},
                )
            else:
                logger.debug(f"Updating labels layer with {selected_labels}")

                self.layer_with_selected_labels.scale = self.scale
                self.layer_with_selected_labels.data = selected_labels

        except TypeError as e:
            show_error(f"Relax filter! {e}")

        self.stat_layer_selector.choices = self.update_labels()[::-1]
        # self.stat_layer_selector.current_choice = "selected labels"

        try:
            self.plot_stats()

        except Exception as e:
            show_error(f"Plot failed: {e}")

    def plot_stats(self, force=False):
        if not self.stat_layer_selector.current_choice:
            return

        if not self.check_auto_plot.value:
            if not force:
                return
        self.pixel_size = float(self.pixel_size_widget.value)
        self.pixel_unit = str(self.pixel_unit_widget.value)
        self.ax[1].set_title(f"{self.diams_title}, {self.pixel_unit}")

        data = self.viewer.layers[self.stat_layer_selector.current_choice].data

        if isinstance(data, dask.array.Array):
            data = data.compute()

        props = [regionprops(label_image=img) for img in data]
        num_regions_per_frame = [len(p) for p in props]
        area_ = [
            [
                (i, prop.area * (self.binning * self.pixel_size) ** 2)
                for prop in props_per_frame
            ]
            for i, props_per_frame in enumerate(props)
        ]
        area = reduce(lambda a, b: a + b, area_[:])

        self.props_df = pd.DataFrame(
            data=area,
            columns=["frame", (area_col := f"area [{self.pixel_unit}$^2$]")],
        )
        self.props_df.loc[:, (diam_col := f"diameter [{self.pixel_unit}]")] = (
            np.sqrt(self.props_df[area_col]) * 2 / np.pi
        )

        eccs_ = [
            [(prop.eccentricity) for prop in props_per_frame]
            for i, props_per_frame in enumerate(props)
        ]
        eccs = reduce(lambda a, b: a + b, eccs_)
        self.props_df.loc[:, (ecc_col := "eccentricity")] = eccs

        data = self.props_df
        self._count.set_data(*zip(*enumerate(num_regions_per_frame)))
        self._diams.set_data(data["frame"], data[diam_col])
        self._eccs.set_data(data["frame"], data[ecc_col])
        [a.set_xlim(0, len(num_regions_per_frame)) for a in self.ax]
        self.ax[0].set_ylim(0, max(num_regions_per_frame) + 1)
        self.ax[1].set_ylim(
            self.props_df[diam_col].min(), self.props_df[diam_col].max()
        )
        self.ax[2].set_ylim(0, 1)

        self.canvas.draw_idle()

    def save_csv(self):
        self.plot_stats(force=True)
        try:
            len(self.props_df)
        except Exception as e:
            logger.error(f"Unable to find the data to save: {e}")
            show_error(f"Unable to find the data to save: {e}")
            return
        path = self.path + ".table.csv"
        self.props_df.loc[:, "filename"] = os.path.basename(self.path)
        self.props_df.loc[:, "layer"] = self.input.current_choice
        try:
            self.props_df.to_csv(path)
            logger.info(f"Saved csv {path}")
            show_info(f"Saved csv {path}")
        except Exception as e:
            logger.error(f"Error saving csv: {e}")
            show_error(f"Error saving csv: {e}")

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
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
        }
        try:

            dir = os.path.dirname(self.path)
            filename = os.path.basename(self.path)
            new_name = filename + ".params.yaml"

            with open(os.path.join(dir, new_name), "w") as f:
                yaml.safe_dump(data, f)
            show_info(f"Parameters saves into {new_name}")
            logger.info(f"Parameters saves into {new_name}")
        except Exception as e:
            show_error(f"Saving parameters into {new_name} failed: {e}")
            logger.error(f"Saving parameters into {new_name} failed: {e}")
        with open((ff := ".latest.params.yaml"), "w") as f:
            yaml.safe_dump(data, f)
            logger.info(f"Parameters saves into {ff}")

    def restore_params(self):
        logger.debug("Start restoring parameters")
        try:
            self.path = self.viewer.layers[self.input.current_choice].metadata[
                "path"
            ]
        except KeyError:
            self.path = ""

        try:
            path = self.path
            dir = os.path.dirname(path)
            filename = os.path.basename(path)
            new_name = filename + ".params.yaml"
        except KeyError:
            pass
        try:
            with open(ppp := os.path.join(dir, new_name)) as f:
                data = yaml.safe_load(f)
            show_info(f"restoring parameters from {new_name}")
            logger.info(f"restoring parameters from {new_name}")
            logger.debug(f"Loaded parameters: {data}")

        except (UnboundLocalError, UnicodeDecodeError, FileNotFoundError):
            try:
                with open(ppp := ".latest.params.yaml") as f:
                    data = yaml.safe_load(f)
                logger.debug(f"Loaded parameters: {data}")
                show_info(f"restoring parameters from {ppp}")
            except FileNotFoundError:
                logger.debug("No latest params found")
        try:
            self.binning_widget.value = data["binning"]
            self.use.value = data["use"]
            self.smooth.value = data["smooth"]
            self.thr.value = data["thr"]
            self.erode.value = data["erode"]
            self.min_diam.value = data["min_diameter"]
            self.max_diam.value = data["max_diameter"]
            self.max_ecc.value = data["max_ecc"]
            self.pixel_size = data["pixel_size"]
            self.pixel_unit = data["pixel_unit"]
        except Exception as e:
            show_error(f"Restore settings failed, {e}")
            logger.error(f"Restore settings failed, {e}")

        self.binning_widget.changed.connect(self.preprocess)
        self.thr.changed.connect(self.threshold)
        self.erode.changed.connect(self.threshold)
        self.use.changed.connect(self.preprocess)
        self.smooth.changed.connect(self.preprocess)
        self.min_diam.changed.connect(self.update_out)
        self.max_diam.changed.connect(self.update_out)
        self.max_ecc.changed.connect(self.update_out)
        self.btn_save_params.clicked.connect(self.save_params)
        self.btn_save_csv.clicked.connect(self.save_csv)

    def reset_choices(self, event=None):
        logger.debug(f"New layer added. Reset choices. Event: {event}")
        new_layers = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image) and layer.name != "Preprocessing"
        ]
        if self.input.choices != new_layers:
            self.input.choices = new_layers
            logger.debug(f"Updating layer list with {new_layers}")
        else:
            logger.debug(
                "No new data layers, probably added pipeline \
                    layers triggered this reset"
            )


def norm01(data):
    d = data.copy()
    if d.max() == 0:
        return d
    return (a := d - d.min()) / a.max()


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


def save_values(val):
    print(f"Saving {val}")


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
    return label(mask)[0].astype("uint16").reshape(labels.shape)


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

    regions = clear_border(regions)

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
