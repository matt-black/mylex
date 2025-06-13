"""Metrics for computing image/volume similarities"""

import jax.numpy as jnp
from jaxtyping import Array

from ._types import Scalar


def psnr(ref: Array, new: Array, mask: Array | None = None) -> Scalar:
    """Compute the peak signal-to-noise (PSNR) between two arrays.

    Args:
        ref (Array): reference array
        new (Array): array to compute PSNR relative to `ref`
        mask (Array | None, optional): Optional foreground mask. PSNR will only be computed where ``mask==1``. Defaults to None.

    Returns:
        Array: _description_
    """
    sq_err = jnp.square(ref - new)
    if mask is not None:
        mask = jnp.broadcast_to(mask, ref.shape)
        mean_sq_err = jnp.sum(sq_err * mask) / jnp.sum(mask).clip(1e-6)
    else:
        mean_sq_err = jnp.mean(sq_err)
    return -10.0 / jnp.log(10.0) * jnp.log(mean_sq_err)
