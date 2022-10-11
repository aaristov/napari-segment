# napari-segment

[![License](https://img.shields.io/pypi/l/napari-segment.svg?color=green)](https://github.com/aaristov/napari-segment/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-segment.svg?color=green)](https://pypi.org/project/napari-segment)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-segment.svg?color=green)](https://python.org)
[![tests](https://github.com/aaristov/napari-segment/workflows/tests/badge.svg)](https://github.com/aaristov/napari-segment/actions)
[![codecov](https://codecov.io/gh/aaristov/napari-segment/branch/main/graph/badge.svg)](https://codecov.io/gh/aaristov/napari-segment)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-segment)](https://napari-hub.org/plugins/napari-segment)

Segment organoids in brightfield from nd2 stack

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.


## Installation

```pip install git+https://github.com/aaristov/napari-segment.git```

## Usage for segmentation

1. Drag your nd2 file into napari
2. Lauch Plugins -> napari-segment: Segment organoid
3. Select the brightfield channel
4. The data is lazily loaded from nd2 dataset and dynamically segmented in the viewer. 
5. Theshold and erode parameters allow you to adjust segmentation -> they all will appear in the Detections layer
6. Min/max diameter and eccentricity allow you to filter out unwanted regions -> the good regions will appear in the "selected labels" layer.
7. You can deactivate the Detection layer with a checkbox.
8. Once saticfied, simply save the selected labels layer with build-in napari saver for future use and downstream analysis.

![image](https://user-images.githubusercontent.com/11408456/176637480-aec8f6f7-d1fe-44dc-b6cd-ccea675c0dc9.png)

## Usage for multicale zarr preview
1. Drag and drop the folder with mutiscale zarr dataset.
2. The plugin will look for the napari attributes in the .zattr file and render the stack accordingly. See the example below for 4D dataset:
```json
{
    "multiscales": {
        "multiscales": [
            {
                "channel_axis": 1,
                "colormap": [
                    "gray",
                    "green",
                    "blue"
                ],
                "datasets": [
                    {
                        "path": "0"
                    },
                    {
                        "path": "1"
                    },
                    {
                        "path": "2"
                    },
                    {
                        "path": "3"
                    }
                ],
                "lut": [
                    [
                        1000,
                        30000
                    ],
                    [
                        440,
                        600
                    ],
                    [
                        0,
                        501
                    ]
                ],
                "name": [
                    "BF",
                    "TRITC",
                    "mask"
                ],
                "title": "BF_TRITC_aligned.zarr",
                "type": "nd2",
                "version": "0.1"
            }
        ]
    }
}
```


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-segment" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/aaristov/napari-segment/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
