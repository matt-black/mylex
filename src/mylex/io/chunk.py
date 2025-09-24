"""Utilities for chunk-wise computation."""

from itertools import product
from typing import Generator

import jax.numpy as jnp
from bioio import BioImage
from jaxtyping import Array

__all__ = [
    "chunk_generator_2d_bioimage",
    "chunk_mesh_2d_props",
    "chunk_generator_1d",
    "assemble_for_pmap",
    "reassemble_from_pmap",
]


def chunk_generator_2d_bioimage(
    im: BioImage,
    n_parts: tuple[int, int],
    overlap: tuple[int, int],
    split_axes: tuple[int, int] = (3, 4),
) -> Generator[tuple[Array, tuple[slice, ...]], None, None]:
    """Create a generator that yields chunks of the specified BioImage.

    Args:
        im (BioImage): image to retrieve chunks of
        n_parts (tuple[int,int]): number of parts (size of mesh) in each dimension
        overlap (tuple[int,int]): amount of overlap to add to each chunk from adjacent chunk.
        split_axes (tuple[int,int]): which axes to split the array over/apply the mesh to.

    Yields:
        Generator[ tuple[Array, tuple[slice,...]], None, None ]: chunks of the array and the slices indicating the part of the input array they correspond to.
    """
    mesh_props = chunk_mesh_2d_props(
        im.data.shape, n_parts, overlap, split_axes
    )
    for chnk, out, pad in mesh_props:
        yield jnp.pad(im.data[chnk], pad, mode="symmetric"), out


def chunk_mesh_2d_props(
    shp: tuple[int, ...],
    n_parts: tuple[int, int],
    overlap: tuple[int, int],
    split_axes: tuple[int, int],
) -> Generator[
    tuple[tuple[slice, ...], tuple[slice, ...], tuple[tuple[int, int], ...]],
    None,
    None,
]:
    """Create a generator that yields tuples of slices indicating how to properly chunk an array of specified shape.

    Args:
        shp (tuple[int,...]): shape of input array to chunk
        n_parts (tuple[int,int]): number of parts (size of mesh) in each dimension
        overlap (tuple[int,int]): amount of overlap to add to each chunk from adjacent chunk.
        split_axes (tuple[int,int]): which axes to split the array over/apply the mesh to.

    Yields:
        Generator[tuple[tuple[slice,...], tuple[slice, ...], tuple[tuple[int,int],...]], None, None]: tuples corresponding to (1) indices to slice into array with, including overlap, (2) indices of the input array the data corresponds to, (3) the amount of padding to add to each chunk
    """
    input_shape = shp
    n_dim = len(input_shape)

    split_shape = [shp[s] // np for s, np in zip(split_axes, n_parts)]

    chunk_slices = []
    out_slices = []
    for cids in product(range(n_parts[0]), range(n_parts[1])):
        this_chnk = [
            slice(
                None,
            )
        ] * n_dim
        this_out = [
            slice(None),
        ] * n_dim
        for i in range(2):
            if cids[i] == 0:
                chunk_slice = slice(0, split_shape[i] + overlap[i])
                out_slice = slice(0, split_shape[i])
            elif cids[i] == n_parts[i] - 1:
                chunk_slice = slice(split_shape[i] * cids[i] - overlap[i], None)
                out_slice = slice(split_shape[i] * cids[i], None)
            else:
                chunk_slice = slice(
                    split_shape[i] * cids[i] - overlap[i],
                    split_shape[i] * (cids[i] + 1) + overlap[i],
                )
                out_slice = slice(
                    split_shape[i] * cids[i], split_shape[i] * (cids[i] + 1)
                )
            this_chnk[split_axes[i]] = chunk_slice
            this_out[split_axes[i]] = out_slice
        chunk_slices.append(tuple(this_chnk))
        out_slices.append(tuple(this_out))

    paddings = []
    for cids in product(range(n_parts[0]), range(n_parts[1])):
        chnk_pad = [
            (0, 0),
        ] * n_dim
        for i in range(2):
            if cids[i] == 0:
                chnk_pad[split_axes[i]] = (overlap[i], 0)
            elif cids[i] == n_parts[i] - 1:
                chnk_pad[split_axes[i]] = (0, overlap[i])
            else:
                pass
        paddings.append(chnk_pad)

    yield from zip(chunk_slices, out_slices, paddings)


def chunk_generator_1d(
    x: Array,
    n_parts: int,
    overlap: int,
    split_axis: int | None,
) -> Generator[Array, None, None]:
    """Create a generator that yields chunks from a one-dimensional mesh of the input array.

    Args:
        x (Array): array to chunk
        n_parts (int): number of parts in the dimension
        overlap (int): amount of overlap to add to each chunk from adjacent chunk.
        split_axes (int|None): which axis to split the array along. if `None`, the array will be split along the largest axis.

    Yields:
        Generator[Array, None, None]: chunks of the array.
    """
    input_shape = x.shape
    n_dim = len(input_shape)
    if split_axis is None:
        split_axis = max(
            [(i, s) for i, s in enumerate(input_shape) if s % n_parts == 0],
            key=lambda t: t[1],
        )[0]
    else:
        if x.shape[split_axis] % n_parts != 0:
            raise ValueError(
                "x's shape on `split_axis` must be evenly divisible by `n_parts`"
            )
    split_shape = x.shape[split_axis] // n_parts

    chunk_slices = []
    for cidx in range(n_parts):
        if cidx == 0:
            chunk_slice = slice(0, split_shape + overlap)
        elif cidx == n_parts - 1:
            chunk_slice = slice(split_shape * cidx - overlap, None)
        else:
            chunk_slice = slice(
                split_shape * cidx - overlap, split_shape * (cidx + 1) + overlap
            )
        this_slice = [
            (chunk_slice if i == split_axis else slice(None))
            for i in range(n_dim)
        ]
        chunk_slices.append(tuple(this_slice))

    paddings = []
    for i in range(n_parts):
        if i == 0:
            padding = [
                ((overlap, 0) if j == split_axis else (0, 0))
                for j in range(n_dim)
            ]
        elif i == n_parts - 1:
            padding = [
                ((0, overlap) if j == split_axis else (0, 0))
                for j in range(n_dim)
            ]
        else:
            padding = [
                (0, 0),
            ] * n_dim
        paddings.append(padding)

    for i in range(n_parts):
        yield jnp.pad(x[chunk_slices[i]], paddings[i], mode="symmetric")


def assemble_for_pmap(
    x: Array,
    n_devices: int,
    overlap: int,
    split_axis: int | None,
) -> Array:
    """Rearrange the array so that chunks of it can be `jax.pmap`'d along its leading axis.

    Args:
        x (Array): input array
        n_devices (int): number of devices
        overlap (int): overlap between each chunk
        split_axis (int | None): axis to split into chunks

    Raises:
        ValueError: if `split_axis` is not evenly divisible by `n_devices`

    Returns:
        Array: input split into subarrays and re-stacked with a leading axis.
    """
    input_shape = x.shape
    n_dim = len(input_shape)
    if split_axis is None:
        split_axis = max(
            [(i, s) for i, s in enumerate(input_shape) if s % n_devices == 0],
            key=lambda t: t[1],
        )[0]
    else:
        if x.shape[split_axis] % n_devices != 0:
            raise ValueError(
                "x's shape on `split_axis` must be evenly divisible by `n_devices`"
            )
    split_shape = x.shape[split_axis] // n_devices

    chunk_slices = []
    for cidx in range(n_devices):
        if cidx == 0:
            chunk_slice = slice(0, split_shape + overlap)
        elif cidx == n_devices - 1:
            chunk_slice = slice(split_shape * cidx - overlap, None)
        else:
            chunk_slice = slice(
                split_shape * cidx - overlap, split_shape * (cidx + 1) + overlap
            )
        this_slice = [
            (chunk_slice if i == split_axis else slice(None))
            for i in range(n_dim)
        ]
        chunk_slices.append(tuple(this_slice))

    paddings = []
    for i in range(n_devices):
        if i == 0:
            padding = [
                ((overlap, 0) if j == split_axis else (0, 0))
                for j in range(n_dim)
            ]
        elif i == n_devices - 1:
            padding = [
                ((0, overlap) if j == split_axis else (0, 0))
                for j in range(n_dim)
            ]
        else:
            padding = [
                (0, 0),
            ] * n_dim
        paddings.append(padding)

    return jnp.stack(
        [
            jnp.pad(x[chunk_slices[i]], paddings[i], mode="symmetric")
            for i in range(n_devices)
        ],
        axis=0,
    )


def reassemble_from_pmap(
    y: Array,
    n_devices: int,
    overlap: int,
    split_axis: int,
) -> Array:
    """Reassemble an array that has been split into chunks by `assemble_for_pmap` into its original shape.

    Args:
        y (Array): input array
        n_devices (int): number of devices that were pmap'd over
        overlap (int): overlap between chunks
        split_axis (int): axis that was split

    Returns:
        Array: reassembled array
    """
    n_dim = len(y.shape) - 1
    slicer = [
        slice(None),
    ] * n_dim
    slicer[split_axis] = slice(overlap, -overlap)
    slicer = tuple(slicer)
    return jnp.concatenate(
        [x[0][slicer] for x in jnp.split(y, n_devices, axis=0)], axis=split_axis
    )
