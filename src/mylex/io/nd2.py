"""Metadata parsing for ND2 files

Convenience functions for getting useful data out of the metadata
"""

import json

import bioio_nd2
from bioio import BioImage, PhysicalPixelSizes


def pixel_size(fpath: str) -> PhysicalPixelSizes:
    """Get the pixel (voxel) size for the ND2 file.

    Args:
        fpath (str): path to ND2 file

    Returns:
        PhysicalPixelSizes: attribute where size can be accessed from `.x`, `.y`, `.z` attributes. Size is in physical units (probably microns).
    """
    im = BioImage(fpath, reader=bioio_nd2.Reader)
    return im.physical_pixel_sizes


def objective_props(fpath: str) -> tuple[float, float]:
    """Get numerical aperture and magnification for objective.

    Args:
        fpath (str): path to ND2 file

    Returns:
        tuple[float,float]: (numerical aperture, magnification)
    """
    im = BioImage(fpath, reader=bioio_nd2.Reader)
    obj = im.metadata.instruments[0].objectives[0]
    na = obj.lens_na
    mag = obj.nominal_magnification
    return na, mag


def zoom(fpath: str) -> float:
    """Get zoom used during acquisition of ND2.

    Args:
        fpath (str): path to ND2 file

    Returns:
        float: zoom
    """
    im = BioImage(fpath, reader=bioio_nd2.Reader)
    d = json.loads(
        im.metadata.structured_annotations.map_annotations[0].value[
            "ImageMetadataSeqLV|0"
        ]
    )["SLxPictureMetadata"]
    return d["Zoom"]
