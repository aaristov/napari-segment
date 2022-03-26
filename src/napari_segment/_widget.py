"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from functools import partial
from multiprocessing import Pool

import napari
import numpy as np
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from scipy.ndimage import label, binary_dilation, gaussian_filter, binary_erosion, binary_fill_holes
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



# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
@magic_factory()
def segment_organoid(
    BF_layer: "napari.layers.Image",
    fluo_layer: "napari.layers.Image",
    donut=10,
    thr: float = 0.4,
) -> napari.types.LayerDataTuple:
    # frame = napari.current_viewer().cursor.position[0]
    kwargs = {}

    try:
        p = Pool()
        stack = p.map(
                partial(seg_frame, thr=thr, donut=donut),
                zip(BF_layer.data.compute(), fluo_layer.data.compute()),
        )
        print((out := np.array(stack)).shape)
        return [(out, {"name": "label", **kwargs}, 'labels')]
        
    except Exception as e:
        print(e.args)
        raise e
    finally:
        p.close()


def seg_frame(data, thr, donut):
    bf, fluo = data
    labels = segment_bf(bf, thr=thr, plot=False)
    print(".",)
    props = regionprops_table(
        labels,
        intensity_image=fluo,
        properties=(
            "label",
            "centroid",
            "bbox",
            "mean_intensity",
            "area",
            "major_axis_length",
        ),
    )
    # print(props)
    biggest_prop_index = np.argmax(props["area"])
    label_of_biggest_object = props["label"][biggest_prop_index]
    spheroid_mask = labels == label_of_biggest_object
    # bg_mask = np.bitwise_xor(
    #     binary_dilation(spheroid_mask, structure=np.ones((donut, donut))),
    #     spheroid_mask,
    # )
    return spheroid_mask.astype("uint16")# + 2 * bg_mask.astype("uint8")


def segment_bf(well, thr=0.2, smooth=10, erode=10, fill=True, plot=False):
    '''
    Serments input 2d array using thresholded gradient with filling
    Returns SegmentedImage object
    '''
    grad = get_2d_gradient(well)
    sm = gaussian_filter(grad, smooth)
#     sm = multiwell.gaussian_filter(well, smooth)
    
    regions = sm > thr * sm.max()
    
    if fill:
        regions = binary_fill_holes(regions)
    
    if erode:
        regions = binary_erosion(regions, iterations=erode)
    labels, _ = label(regions)
    
    return labels

def get_2d_gradient(xy):
    gx, gy = np.gradient(xy)
    return np.sqrt(gx ** 2 + gy ** 2)