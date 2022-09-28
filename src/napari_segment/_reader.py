"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/stable/guides.html#readers
"""
import json
import os

import dask
import nd2
import numpy as np
import tifffile as tf


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if path.endswith(".nd2"):
        return read_nd2

    if path.endswith(".zarr"):
        return read_zarr

    if path.endswith(".tif"):
        return read_tif

    # otherwise we return the *function* that can read ``path``.
    if path.endswith(".npy"):
        return reader_function

    return None


def read_tif(path):
    data = tf.TiffFile(path)
    arr = data.asarray()
    channel_axis = (
        arr.shape.index(data.imagej_metadata["channels"])
        if data.is_imagej
        else None
    )

    return [
        (
            arr,
            {"channel_axis": channel_axis, "metadata": {"path": path}},
            "image",
        )
    ]


def read_zarr(path):
    print(f"read_zarr {path}")

    try:
        attrs = json.load(open(os.path.join(path, ".zattrs")))
        info = attrs["multiscales"]["multiscales"][0]
    except Exception as e:
        raise e

    dataset_paths = [os.path.join(path, d["path"]) for d in info["datasets"]]
    datasets = [dask.array.from_zarr(p) for p in dataset_paths]

    try:
        channel_axis = info["channel_axis"]
        print(f"found channel axis {channel_axis}")
    except KeyError:
        channel_axis = None

    except Exception as e:
        raise e

    try:
        contrast_limits = info["lut"]
    except KeyError:
        contrast_limits = None

    try:
        colormap = info["colormap"]
    except KeyError:
        colormap = None

    try:
        name = info["name"]
    except KeyError:
        print("name not found")
        name = [os.path.basename(path)] * datasets[0].shape[channel_axis]
    except Exception as e:
        print("name exception", e.args)
        name = os.path.basename(path)

    return [
        (
            datasets,
            {
                "channel_axis": channel_axis,
                "colormap": colormap,
                "contrast_limits": contrast_limits,
                "name": name,
            },
            "image",
        )
    ]


def read_nd2(path):
    print(f"reading {path}")
    data = nd2.ND2File(path)
    print(data.sizes)
    ddata = data.to_dask()
    # colormap = ["gray", "green"]
    try:
        channel_axis = list(data.sizes.keys()).index("C")
    except ValueError:
        print(f"No channels, {data.sizes}")
        channel_axis = None
        # colormap = ["gray"]
    return [
        (
            ddata,
            {
                "channel_axis": channel_axis,
                "metadata": {"sizes": data.sizes, "path": path},
            },
            # dict(
            #     channel_axis=channel_axis,
            #     name=[ch.channel.name for ch in data.metadata.channels],
            # colormap=colormap,
            # scale=data.metadata.channels[0].volume.axesCalibration[:]
            # contrast_limits=[(8500, 35000), (150, 20000)],
            # ),
            "image",
        )
    ]


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]
