"""Fourier ring correlation"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm.auto import trange

__all__ = ["single_image_frc"]


def single_image_frc(
    im: Float[Array, "r c"] | Int[Array, "r c"],
    bin_width: int,
    n_trials: int,
    key: PRNGKeyArray,
    pixel_size: float = 1.0,
    verbose: bool = False,
) -> tuple[Float[Array, " k"], Float[Array, "k {n_trials}"]]:
    """Compute the single image FRC for the input image.

    Args:
        im (Float[Array, "r c"]|Int[Array, "r c"]): input image
        bin_width (int): width of radial bin, in pixels
        n_trials (int): number of trials to compute
        key (PRNGKeyArray): prng key
        pixel_size (float): size of pixels, in physical units. Defaults to 1.0
        verbose (bool): show a progress bar during trials. Defaults to False.

    Returns:
        tuple[Float[Array, " k"], Float[Array, " k"]]: (frequency, )
    """
    key, apply_key = jr.split(key, 2)
    im1, im2 = coinflip_split(im, apply_key)
    rad, val = _single_image_frc_trial(im1, im2, bin_width, pixel_size)
    vals = [val]
    for _ in (
        trange(n_trials, desc="FRC Trials") if verbose else range(n_trials)
    ):
        key, apply_key = jr.split(key, 2)
        im1, im2 = coinflip_split(im, apply_key)
        _, val = _single_image_frc_trial(im1, im2, bin_width)
        vals.append(val)
    vals = jnp.stack(vals, axis=0)
    return rad, vals


def _single_image_frc_trial(
    im1: Float[Array, "r c"] | Int[Array, "r c"],
    im2: Float[Array, "r c"] | Int[Array, "r c"],
    bin_width: int,
    pixel_size: float = 1.0,
) -> tuple[Float[Array, " k"], Float[Array, " k"]]:
    # coinflip split input image
    # fft the two images
    g1 = jnp.fft.fftshift(jnp.fft.fft2(im1))
    g2 = jnp.fft.fftshift(jnp.fft.fft2(im2))
    # figure out frequencies
    frq = jnp.fft.fftshift(jnp.fft.fftfreq(min(im1.shape), pixel_size))

    # compute terms we need later
    xcorr = jnp.real(jnp.multiply(g1, jnp.conj(g2))).flatten()
    g1mag = jnp.square(jnp.abs(g1)).flatten()
    g2mag = jnp.square(jnp.abs(g2)).flatten()
    # make radius image, then get keys for how to sort it
    radim = _radii_image(im1).flatten()
    key = jnp.argsort(radim)
    # sort all of the vectors by radii key
    radim, xcorr, g1mag, g2mag = radim[key], xcorr[key], g1mag[key], g2mag[key]
    radii = jnp.arange(bin_width, min(im1.shape) / 2, 1)
    frq = frq[-len(radii) :]

    def _get_bounds(rad: Float) -> tuple[Int, Int]:
        idxs = jnp.nonzero(
            jnp.logical_and(radim >= rad - bin_width, radim < rad + bin_width),
            size=len(radim) // 4,
            fill_value=-1,
        )[0]
        min_idx = jnp.amin(idxs, initial=radim.shape[0], where=idxs >= 0)
        max_idx = jnp.amax(idxs)
        return min_idx, max_idx

    def _do_averaging(y: Array, min_idx: Int, max_idx: Int) -> Float:
        arr = jax.lax.slice_in_dim(y, min_idx, max_idx, 1, 0)
        return jnp.mean(arr)

    mins, maxs = jax.vmap(_get_bounds, 0, 0)(radii)
    funs = [
        Partial(_do_averaging, min_idx=m1, max_idx=m2)
        for m1, m2 in zip(mins, maxs)
    ]
    # vals is like (# radii x 3) where 3 is: (xcorr, g1mag, g2mag)
    vals = jnp.stack(
        [
            jax.vmap(f, 0, 0)(jnp.stack([xcorr, g1mag, g2mag], axis=0))
            for f in funs
        ],
        axis=0,
    )
    frc = vals[:, 0] / jnp.sqrt(vals[:, 1] * vals[:, 2])
    return frq, frc


def _radii_image(
    im: Float[Array, "r c"] | Int[Array, "r c"],
) -> Float[Array, "y x"]:
    d0, d1 = im.shape
    c0, c1 = d0 / 2, d1 / 2
    x, y = jnp.meshgrid(
        jnp.linspace(-c1, c1, num=d1, endpoint=True),
        jnp.linspace(-c0, c0, num=d0, endpoint=True),
        indexing="xy",
    )
    return jnp.sqrt(jnp.add(jnp.square(x), jnp.square(y)))


def coinflip_split(arr: Array, key: PRNGKeyArray) -> tuple[Array, Array]:
    """Split an image into 2 parts for FRC processing by "coin flip" splitting the "photons" at each phixel.

    Args:
        arr (Array): array to split.
        key (PRNGKeyArray): PRNG key.

    Returns:
        tuple[Array, Array]: splits of the input array.
    """
    n1 = jr.binomial(key, arr, 0.5)
    n2 = arr - n1
    return n1, n2


def highfreq_plateau(
    gain_err: float,
    offset_err: float,
    rms_readout_noise: float,
    exp_photon_count_pixel: float,
) -> float:
    """Calculate the value of the high-frequency plateau for single image FRC.

    Args:
        gain_err (float): error in determining gain
        offset_err (float): error in determining offset
        rms_readout_noise (float): RMS readout noise
        exp_photon_count_pixel (float): expected per-pixel photon count

    Returns:
        float: plateau value
    """
    return (
        gain_err * (gain_err - 1)
        - offset_err / exp_photon_count_pixel
        + gain_err**2 * rms_readout_noise**2 / exp_photon_count_pixel
    ) / (
        gain_err * (gain_err + 1)
        + offset_err / exp_photon_count_pixel
        + gain_err**2 * rms_readout_noise**2 / exp_photon_count_pixel
    )
