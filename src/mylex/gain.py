"""Gain estimation

Implementation of the approach described in Heintzmann, et al. [1]

References
---
[1] Heintzmann, R., Relich, P. K., Nieuwenhuizen, R. P., Lidke, K. A., & Rieger, B. (2016). Calibrating photon counts from a single image. arXiv preprint arXiv:1611.05654.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from bioio import BioImage
from bioio_nd2 import Reader as nd2_Reader
from bioio_tifffile import Reader as tiff_Reader
from jaxtyping import Array, Bool, Complex, Float

from .io.chunk import chunk_generator_2d_bioimage


def fft_symmetrized(x: Float[Array, "..."]) -> Complex[Array, "..."]:
    """Symmetrized, centered fourier transform of the input where the input has been symmetrized to prevent boundary artifacts in the transform.

    Args:
        x (Float[Array, "..."]): input array

    Returns:
        Complex[Array, "..."]: fft of the input, shifted so that 0 frequency is in the center.
    """
    pad_width = tuple([(0, d) for d in x.shape])
    start_ind = tuple([d // 2 for d in x.shape])
    symm = jnp.pad(x, pad_width, mode="symmetric")
    fft = jnp.fft.fftshift(jnp.fft.fftn(symm))
    return jax.lax.dynamic_slice(fft, start_ind, x.shape)


@partial(jax.jit, static_argnums=(1, 2))
def estimate_mean_noise_energy(
    arr: Array, k_t: float, sample_spacing: float = 1.0
) -> Float[Array, ""]:
    """Estimate the mean noise energy by averaging the power of all the pixels in the FFT above the cutoff frequency, `k_t`.

    Args:
        arr (Array): array to estimate mean noise energy of
        k_t (float): cutoff frequency
        sample_spacing (float): scalar sample spacing (e.g. pixel size, voxel size). determines the units of the cutoff frequency.

    Returns:
        Float[Array, ""]: scalar mean noise energy
    """
    frq = jnp.fft.fftfreq(min(arr.shape), d=sample_spacing)
    fft = jnp.fft.fftshift(jnp.fft.fftn(arr))
    pwr = jnp.square(jnp.abs(fft))
    frq_grid = jnp.stack(
        jnp.meshgrid(
            *(
                [
                    frq,
                ]
                * arr.ndim
            ),
            indexing="ij",
        ),
        axis=0,
    )
    frq_rad = jnp.sqrt(jnp.sum(jnp.square(frq_grid), axis=0))
    frq_msk = frq_rad > k_t
    T = jnp.sum(frq_msk)
    return jnp.divide(jnp.sum(pwr[frq_msk]), T)


def high_freq_mask_2d(
    size_y: int,
    size_x: int,
    pix_size: float,
    res_sigma: float,
) -> Bool[Array, "{y_size} {x_size}"]:
    abv_y = jnp.divide(1.0, _freqs_for_dim(size_y, pix_size)) < res_sigma
    abv_x = jnp.divide(1.0, _freqs_for_dim(size_x, pix_size)) < res_sigma
    return jnp.einsum("i,j->ij", abv_y, abv_x)


def high_freq_mask_3d(
    size_z: int,
    size_y: int,
    size_x: int,
    vox_size_ax: float,
    vox_size_lat: float,
    ax_res_sigma: float,
    lat_res_sigma: float,
) -> Bool[Array, "{z_size} {y_size} {x_size}"]:
    abv_ax = jnp.divide(1.0, _freqs_for_dim(size_z, vox_size_ax)) < ax_res_sigma
    abv_y = (
        jnp.divide(1.0, _freqs_for_dim(size_y, vox_size_lat)) < lat_res_sigma
    )
    abv_x = (
        jnp.divide(1.0, _freqs_for_dim(size_x, vox_size_lat)) < lat_res_sigma
    )
    return jnp.einsum("i,j,k->ijk", abv_ax, abv_y, abv_x)


def _freqs_for_dim(shp: int, s: float) -> Float[Array, " {shp}"]:
    return jnp.abs(jnp.fft.fftshift(jnp.fft.fftfreq(shp, s)))


def mean_variance_curve_for_volume(
    fpath: str,
    mask: Bool[Array, "z y x"],
    channel: int,
    mesh_shape: tuple[int, int],
    overlap: tuple[int, int],
    split_axes: tuple[int, int] = (3, 4),
    time: int = 0,
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    if fpath.endswith("nd2"):
        bio_img = BioImage(fpath, reader=nd2_Reader)
    elif fpath.endswith("tif") or fpath.endswith("tiff"):
        bio_img = BioImage(fpath, reader=tiff_Reader)
    else:
        raise ValueError("only nd2s and tiffs are supported")
    chunks = chunk_generator_2d_bioimage(
        bio_img, mesh_shape, overlap, split_axes
    )
    T = jnp.sum(mask)
    means, vars = [], []
    for chunk, _ in chunks:
        chunk = chunk[time, channel, ...]
        c_sel = chunk[:, overlap[0] : -overlap[0], overlap[1] : -overlap[1]]
        oz, _, _ = chunk.shape
        # symmetrize z-dimension by padding
        chunk = jnp.pad(
            chunk, ((oz // 2, oz // 2), (0, 0), (0, 0)), mode="symmetric"
        )
        c_fft = jnp.fft.fftshift(jnp.fft.fftn(chunk)) / math.sqrt(
            math.prod(chunk.shape)
        )
        c_fft = c_fft[
            oz // 2 : -oz // 2 + 1,
            overlap[0] : -overlap[0],
            overlap[1] : -overlap[1],
        ]
        c_pwr = jnp.abs(jnp.square(c_fft))
        M = math.prod(c_sel.shape)
        avg_pix = jnp.sum(c_sel) / M
        avg_pow = jnp.sum(c_pwr[mask]) / T
        means.append(avg_pix)
        vars.append(avg_pow)
    means = jnp.asarray(means)
    vars = jnp.asarray(vars)
    return means, vars


def estimate_zeronoise_offset(
    mean: Float[Array, " n"], variance: Float[Array, " n"]
) -> Float[Array, ""]:
    # do poisson-weighted regression
    X = jnp.vstack([variance, jnp.ones_like(mean)]).transpose()
    Y = mean
    Wi = jnp.diag(1.0 / jnp.square(variance))
    par = jnp.linalg.solve(
        X.T @ Wi @ X, X.T @ Wi @ Y
    )  # output is (slope, y-intercept)
    # only want the y-intercept
    return par


def estimate_gain_chunkwise3d(
    fpath: str,
    zeronoise_offset: Float[Array, ""] | float,
    mask: Bool[Array, "z y x"],
    channel: int,
    mesh_shape: tuple[int, int],
    overlap: tuple[int, int],
    split_axes: tuple[int, int] = (3, 4),
    time: int = 0,
) -> Float[Array, ""]:
    if fpath.endswith("nd2"):
        bio_img = BioImage(fpath, reader=nd2_Reader)
    elif fpath.endswith("tif") or fpath.endswith("tiff"):
        bio_img = BioImage(fpath, reader=tiff_Reader)
    else:
        raise ValueError("only nd2s and tiffs are supported")
    chunks = chunk_generator_2d_bioimage(
        bio_img, mesh_shape, overlap, split_axes
    )
    T = jnp.sum(mask)
    gains = []
    for chunk, _ in chunks:
        chunk = chunk[time, channel, ...]
        c_sel = chunk[:, overlap[0] : -overlap[0], overlap[1] : -overlap[1]]
        oz, _, _ = chunk.shape
        # symmetrize z-dimension by padding
        chunk = jnp.pad(
            chunk, ((oz // 2, oz // 2), (0, 0), (0, 0)), mode="symmetric"
        )
        c_fft = jnp.fft.fftshift(jnp.fft.fftn(chunk)) / math.sqrt(
            math.prod(chunk.shape)
        )
        c_fft = c_fft[
            oz // 2 : -oz // 2 + 1,
            overlap[0] : -overlap[0],
            overlap[1] : -overlap[1],
        ]
        c_pwr = jnp.abs(jnp.square(c_fft))
        M = math.prod(c_sel.shape)
        chunk_gain = jnp.divide(
            M * jnp.sum(c_pwr[mask]), T * jnp.sum(c_sel - zeronoise_offset)
        )
        gains.append(chunk_gain)
    return jnp.asarray(gains)
