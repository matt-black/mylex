"""Adaptive median filtering

Like a median filter, but instead of replacing each pixel with the median of its local neighborhood, only do the replacement if its value exceeds some multiplicative threshold of the median.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array
from winx.ndimage import generic_filter, median_filter

__all__ = [
    "adaptive_median_filter",
    "median_filter",
]


def adaptive_median_filter(
    x: Array,
    window_size: int,
    threshold: float,
) -> Array:
    """Apply an adaptive median filter to the array.

    Algorithm is described in [1].

    Args:
        x (Array): input array to be filtered
        window_size (int): size of window (will be square/cubic/etc. based on dimensionality of array)
        threshold (float): multiplicative threshold for determining hot pixels/voxels to replace.

    Returns:
        Array: filtered array

    References:
        [1] Weisong Zhao et al. Sparse deconvolution improves the resolution of live-cell super-resolution fluorescence microscopy, Nature Biotechnology 40, 606-617 (2022).
    """
    c = jnp.array([window_size // 2 for _ in x.shape])
    sel = Partial(
        jax.lax.dynamic_slice, start_indices=c, slice_sizes=[1 for _ in x.shape]
    )
    fun = Partial(amf_window_function, threshold=threshold, sel_fun=sel)
    return generic_filter(
        x,
        fun,
        window_size,
        footprint=None,
        mode="symmetric",
    )


@Partial(jax.jit, static_argnums=(1, 2))
def amf_window_function(
    window: Array, threshold: float, sel_fun: Callable[[Array], Array]
) -> Array:
    median = jnp.median(window)
    val = sel_fun(window)
    return jnp.where(val > threshold * median, median, val)
