from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._types import Scalar


@jax.jit
def _select_masked_pixels(
    data: Array, pred: Array, mask: Array
) -> Tuple[Array, Array, Scalar]:
    """Select only the pixels in data and pred inputs that have value 1 at the corresponding coordinate in the input mask.

    Args:
        data (Array): data array
        pred (Array): predictions array
        mask (Array): mask array

    Returns:
        Tuple[Array, Array]: subsets of (data, pred) arrays
    """
    # unmask is one where the data was masked out
    # zero where the data was not masked out
    unmask = (1 - mask) > 0.5
    n_pix = jnp.sum(unmask)
    sel_data = jnp.where(unmask, data, 0)
    sel_pred = jnp.where(unmask, pred, 0)
    return sel_data, sel_pred, n_pix


@jax.jit
def l2(data: Array, pred: Array) -> Scalar:
    return jnp.sqrt(jnp.sum(jnp.square(data - pred)))


@jax.jit
def l1(data: Array, pred: Array) -> Scalar:
    return jnp.abs(data - pred)


@jax.jit
def loss_s2s(data: Array, pred: Array, mask: Array) -> Scalar:
    """Loss function used in the Self2Self paper. This is just the L2 loss on the pixels that were masked out during the forward pass.

    Args:
        data (Array): data array
        pred (Array): predictions array
        mask (Array): mask array (should be that used to originally Bernoulli mask data)

    Raises:
        ValueError: if invalid distance is specified

    Returns:
        Scalar
    """
    sel_data, sel_pred, num_pix = _select_masked_pixels(data, pred, mask)
    return l2(sel_data, sel_pred) / num_pix
    # return jax.lax.cond(use_l1, l1, l2, sel_data, sel_pred) / num_pix
