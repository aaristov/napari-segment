try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._reader import napari_get_reader
from ._sample_data import make_early_aggregate, make_late_aggregate
from ._widget import SegmentStack, example_magic_widget
from ._writer import write_multiple, write_single_image

__ALL__ = [
    napari_get_reader,
    write_multiple,
    write_single_image,
    make_early_aggregate,
    make_late_aggregate,
    SegmentStack,
    example_magic_widget,
]
