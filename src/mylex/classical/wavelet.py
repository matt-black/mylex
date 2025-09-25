"""Denoising functions"""

from typing import Sequence, Union

import jax.numpy as jnp
from calcite.dwt import wavedec, waverec
from calcite.wavelet.discrete import DiscreteWavelet
from jax.tree_util import Partial
from jaxtyping import Array

from ..gauss import gaussian


def wavelet_based_bgnoise_estimate(
    x: Array,
    n_bg_levels: int,
    noise_level: int,
    wavelets: (
        str | DiscreteWavelet | Sequence[Union[str, DiscreteWavelet]]
    ) = "db1",
) -> tuple[Array, Array]:
    """Perform wavelet-based background and noise estimation, using the method described in [1].

    Args:
        x (Array): array to have background & noise of estimated.
        n_bg_levels (int): number of levels to approximate as background
        noise_level (int): highest level considered noise (lower numbers represent higher frequencies).
        wavelets (str | DiscreteWavelet | Sequence[Union[str, DiscreteWavelet]], optional): wavelets to use. Defaults to 'db1'.

    Returns:
        tuple[Array, Array]: tuple of estimated (background, noise).

    References:
        [1]  Manuel Hüpfel, Andrei Yu. Kobitski, Weichun Zhang, and G. Ulrich Nienhaus, "Wavelet-based background and noise subtraction for fluorescence microscopy images," Biomed. Opt. Express 12, 969-980 (2021)

    Notes:
        If one wants to estimate both background and noise, this function will only do the wavelet decomposition once, saving computation time compared to called `wavelet_based_background_estimate` and `wavelet_based_noise_estimate` separately. But it will keep two copies of the detail coefficients in memory during background estimation, leading to higher memory usage.
    """
    approx, coeffs = wavedec(x, wavelets, level=None)
    # background identification & reconstruction
    # zero out all the detail levels below `n_levels`, which will only leave the background
    bg_coeffs = []
    for lvl in range(len(coeffs)):
        if lvl < n_bg_levels:  # zero out the detail coefficients at this level
            _coeffs = []
            for coeff in coeffs[lvl]:
                _coeffs.append(jnp.zeros_like(coeff))
            bg_coeffs.append(_coeffs)
        else:  # do nothing
            bg_coeffs.append(coeffs[lvl])
    # smooth out by gaussian filtering, with sigma determined by # of levels
    bg_rec = waverec(approx, coeffs, wavelets)
    bg = _blur_background(bg_rec, n_bg_levels, 3)
    approx = jnp.ones_like(approx)
    for lvl in range(noise_level, len(coeffs), 1):
        coeffs[lvl] = [jnp.zeros_like(c) for c in coeffs[lvl]]
    noise = waverec(approx, coeffs, wavelets)
    return bg, noise


def wavelet_based_background_estimate(
    x: Array,
    n_levels: int,
    wavelets: (
        str | DiscreteWavelet | Sequence[Union[str, DiscreteWavelet]]
    ) = "db1",
    max_decomp_level: int | None = None,
) -> Array:
    """Perform wavelet-based background estimation, using the method described in [1].

    Args:
        x (Array): array to have background subtracted & noise estimated.
        n_levels (int): number of levels to approx. as background.
        wavelets (str | DiscreteWavelet | Sequence[Union[str, DiscreteWavelet]], optional): wavelets to use. Defaults to 'db1'.
        estimate_noise (bool, optional): estimate the noise

    Returns:
        Array

    References:
        [1]  Manuel Hüpfel, Andrei Yu. Kobitski, Weichun Zhang, and G. Ulrich Nienhaus, "Wavelet-based background and noise subtraction for fluorescence microscopy images," Biomed. Opt. Express 12, 969-980 (2021)
    """
    approx, coeffs = wavedec(x, wavelets, level=max_decomp_level)

    # background identification & reconstruction
    # zero out all the detail levels below `n_levels`, which will only leave the background
    for lvl in range(len(coeffs)):
        if lvl < n_levels:  # zero out the detail coefficients at this level
            _coeffs = []
            for coeff in coeffs[lvl]:
                _coeffs.append(jnp.zeros_like(coeff))
            coeffs[lvl] = _coeffs
        else:  # do nothing
            pass
    # smooth out by gaussian filtering, with sigma determined by # of levels
    recon_odd = tuple([s % 2 > 0 for s in x.shape])
    bg_rec = waverec(approx, coeffs, wavelets, recon_odd=recon_odd)
    bg = _blur_background(bg_rec, n_levels, 3)
    return bg


def wavelet_based_noise_estimate(
    x: Array,
    level: int,
    wavelets: (
        str | DiscreteWavelet | Sequence[Union[str, DiscreteWavelet]]
    ) = "db1",
) -> Array:
    """Perform wavelet-based noise estimation, using the method described in [1].

    Args:
        x (Array): array to have background subtracted & noise estimated.
        level (int): decomposition level representing high frequency noise.
        wavelets (str | DiscreteWavelet | Sequence[Union[str, DiscreteWavelet]], optional): wavelets to use. Defaults to 'db1'.
        estimate_noise (bool, optional): estimate the noise

    Returns:
        Array

    References:
        [1]  Manuel Hüpfel, Andrei Yu. Kobitski, Weichun Zhang, and G. Ulrich Nienhaus, "Wavelet-based background and noise subtraction for fluorescence microscopy images," Biomed. Opt. Express 12, 969-980 (2021)
    """
    recon_odd = tuple([s % 2 > 0 for s in x.shape])
    approx, coeffs = wavedec(x, wavelets, level=None)
    approx = jnp.ones_like(approx)
    for lvl in range(level, len(coeffs), 1):
        coeffs[lvl] = [jnp.zeros_like(c) for c in coeffs[lvl]]
    return waverec(approx, coeffs, wavelets, recon_odd=recon_odd)


def _blur_background(
    bkgrnd_rec: Array, n_levels: int, trunc: float = 3.0
) -> Array:
    sigma = 2**n_levels
    blur_kern_size = min(2**n_levels * trunc, min(bkgrnd_rec.shape))
    x = jnp.linspace(-blur_kern_size, blur_kern_size, 1 + 2 * blur_kern_size)
    blur_kern = gaussian(x, sigma)
    conv_fun = Partial(jnp.convolve, v=blur_kern, mode="same")

    def conv_axis(idx: int, val: Array):
        return jnp.apply_along_axis(conv_fun, idx, val)

    for ax in range(bkgrnd_rec.ndim):
        bkgrnd_rec = conv_axis(ax, bkgrnd_rec)
    return bkgrnd_rec
