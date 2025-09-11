import math

import jax.numpy as jnp
from jaxtyping import Array


def normalize_0_to_1(x: Array) -> Array:
    min_val = jnp.min(x)
    max_val = jnp.max(x)
    return (x - min_val) / (max_val - min_val)


def fwhm_to_sigma(fwhm: float) -> float:
    """Convert the full-width at half maximum value of a Gaussian to standard deviation.

    Args:
        fwhm (float): full width at half maximum

    Returns:
        float: standard deviation
    """
    return fwhm / (2 * math.sqrt(2 * math.log(2)))


def sigma_to_fwhm(sigma: float) -> float:
    """Convert the standard deviation of a Gaussian to full width half maximum.

    Args:
        fwhm (float): standard deviation

    Returns:
        float: full width at half maximum
    """
    return 2 * math.sqrt(2 * math.log(2)) * sigma
