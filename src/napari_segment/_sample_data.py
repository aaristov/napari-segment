"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/stable/guides.html#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import os
import pathlib

from ._reader import read_nd2

URL = "https://github.com/aaristov/napari-segment/releases/download/v0.2.8/"


DATA = [
    ("D3_D1.nd2", read_nd2),
    ("D3_D4.nd2", read_nd2),
]

PARAMS = [
    ("D3_D1.nd2.params.yaml", None),
    ("D3_D4.nd2.params.yaml", None),
]

SAMPLE_FOLDER = ".napari-segment"


def make_early_aggregate():
    data = _load_sample_data(
        *DATA[0],
        name="D3_D1",
        colormap="gray",
        opacity=1,
    )
    _ = _load_sample_data(*PARAMS[0])
    return data


def make_late_aggregate():
    data = _load_sample_data(
        *DATA[1],
        name="D3_D4",
        colormap="gray",
        opacity=1,
    )
    _ = _load_sample_data(*PARAMS[1])
    return data


def download_url_to_file(
    url,
    file_path,
):
    import shutil

    import urllib3

    print(f"Downloading {url}")
    c = urllib3.PoolManager()
    with c.request("GET", url, preload_content=False) as resp, open(
        file_path, "wb"
    ) as out_file:
        shutil.copyfileobj(resp, out_file)
    resp.release_conn()
    print(f"Saved {file_path}")
    return file_path


def _load_sample_data(image_name, readfun=read_nd2, **kwargs):

    cp_dir = pathlib.Path.home().joinpath(SAMPLE_FOLDER)
    cp_dir.mkdir(exist_ok=True)
    data_dir = cp_dir.joinpath("data")
    data_dir.mkdir(exist_ok=True)

    url = URL + image_name

    cached_file = str(data_dir.joinpath(image_name))
    if not os.path.exists(cached_file):
        print(f"Downloading {image_name}")
        download_url_to_file(url, cached_file)

    if readfun is None:
        return

    return readfun(cached_file, **kwargs)
