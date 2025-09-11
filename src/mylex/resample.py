import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    "additive_downsample",
    "fourier_downsample",
    "fourier_upsample",
]


def additive_downsample(x: Array, ratio: int) -> Array:
    """Downsample the input image by the specified ratio by summing local neighborhoods.

    Args:
        x (Array): array to downsample
        ratio (int): ratio to downsample array by (width of convolution kernel)

    Returns:
        Array: downsampled array
    """
    n_dim = len(x.shape)
    rhs = jnp.ones(
        [
            ratio,
        ]
        * n_dim,
        dtype=x.dtype,
    )[None, None, ...]
    return jax.lax.conv_general_dilated(
        x[None, None, ...],
        rhs,
        [
            ratio,
        ]
        * n_dim,
        "valid",
    )[0, 0, ...]


def fourier_downsample(x: Array, target_shape: tuple[int, ...]) -> Array:
    """Downsample the input image or volume to the specified shape by truncation of the Fourier transform.

    Args:
        x (Array): array to be downsampled.
        target_shape (tuple[int, ...]): shape to downsample the array to.

    Returns:
        Array
    """
    cent = [s // 2 for s in x.shape]
    rad = [s // 2 for s in target_shape]
    odd = [int(s % 2) for s in target_shape]
    slices = [slice(c - r, c + r + o) for c, r, o in zip(cent, rad, odd)]
    return jnp.fft.ifftn(
        jnp.fft.ifftshift(jnp.fft.fftshift(jnp.fft.fftn(x))[*slices])
    ).real


def fourier_upsample(
    x: Float[Array, "y x"] | Float[Array, "z y x"],
    target_shape: tuple[int, int] | tuple[int, int, int],
) -> Float[Array, "y1 x1"] | Float[Array, "z1 y1 x1"]:
    """Upsample an axis by Fourier upsampling, as described in [1].

    Args:
        x (Float[Array, "y x"]|Float[Array, "z y x"]): input array (2- or 3-D) to upsample.
        target_shape (tuple[int,int]|tuple[int,int,int]): shape to upsample to.

    Returns:
        Float[Array, "y x"]|Float[Array, "z y x"]: upsampled image or volume.

    References:
        [1] Simon C. Stein, Anja Huss, Dirk Hähnel, Ingo Gregor, and Jörg Enderlein, "Fourier interpolation stochastic optical fluctuation imaging," Opt. Express 23, 16154-16163 (2015).
    """
    if not len(target_shape) == x.ndim:
        raise ValueError(
            "target shape doesnt match dimensionality of input data"
        )
    if x.ndim == 2:
        return _fourier_upsample_2d(x, (target_shape[0], target_shape[1]))
    elif x.ndim == 3:
        return _fourier_upsample_3d(
            x,
            (
                target_shape[0],
                target_shape[1],
                target_shape[2],  # pyright: ignore[reportGeneralTypeIssues]
            ),
        )
    else:
        raise ValueError("only works on 2- and 3D arrays")


def _fourier_upsample_2d(
    x: Float[Array, "y0 x0"],
    target_shape: tuple[int, int],
) -> Float[Array, "y1 x1"]:
    sze_y, sze_x = x.shape
    rat_y, rat_x = target_shape[0] / sze_y, target_shape[1] / sze_x
    if rat_y < 1.0 or rat_x < 1.0:
        raise ValueError("target must be >= current shape in all dimensions")
    symm_pad = ((sze_y // 2, sze_y // 2), (sze_x // 2, sze_x // 2))
    fourier = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.pad(x, symm_pad, mode="symmetric"))
    )
    n12_y = math.ceil((2 * sze_y - 1) / 2)
    delta_y = int(math.ceil(rat_y * n12_y - n12_y))
    n12_x = math.ceil((2 * sze_x - 1) / 2)
    delta_x = int(math.ceil(rat_x * n12_x - n12_x))
    # pad fourier domain image with zeros
    fourier = jnp.pad(
        fourier,
        ((delta_y, delta_y), (delta_x, delta_x)),
        mode="constant",
        constant_values=0,
    )
    # do ifft
    out = jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(fourier)))
    # trim out extra stuff generate by the padding
    n_mid_y = math.floor(rat_y * sze_y)
    nmh_y = n_mid_y // 2
    n_mid_x = math.floor(rat_x * sze_x)
    nmh_x = n_mid_x // 2
    cnt_y, cnt_x = out.shape[0] // 2, out.shape[1] // 2
    return out[cnt_y - nmh_y : cnt_y + nmh_y, cnt_x - nmh_x : cnt_x + nmh_x]


def _fourier_upsample_3d(
    x: Float[Array, "z0 y0 x0"],
    target_shape: tuple[int, int, int],
) -> Float[Array, "z1 y1 x1"]:
    sze_z, sze_y, sze_x = x.shape
    rat_z, rat_y, rat_x = (
        target_shape[0] / sze_z,
        target_shape[1] / sze_y,
        target_shape[2] / sze_x,
    )
    if rat_z < 1.0 or rat_y < 1.0 or rat_x < 1.0:
        raise ValueError("target must be >= current shape in all dimensions")
    symm_pad = (
        (sze_z // 2, sze_z // 2),
        (sze_y // 2, sze_y // 2),
        (sze_x // 2, sze_x // 2),
    )
    fourier = jnp.fft.fftshift(
        jnp.fft.fftn(jnp.pad(x, symm_pad, mode="symmetric"))
    )
    n12_z = math.ceil((2 * sze_z - 1) / 2)
    delta_z = int(math.ceil(rat_y * n12_z - n12_z))
    n12_y = math.ceil((2 * sze_y - 1) / 2)
    delta_y = int(math.ceil(rat_y * n12_y - n12_y))
    n12_x = math.ceil((2 * sze_x - 1) / 2)
    delta_x = int(math.ceil(rat_x * n12_x - n12_x))
    # pad fourier domain image with zeros
    fourier = jnp.pad(
        fourier,
        ((delta_z, delta_z), (delta_y, delta_y), (delta_x, delta_x)),
        mode="constant",
        constant_values=0,
    )
    # do ifft
    out = jnp.fft.ifftn(jnp.fft.ifftshift(fourier))
    # trim out extra stuff generate by the padding
    n_mid_z = math.floor(rat_z * sze_z)
    nmh_z = n_mid_z // 2
    n_mid_y = math.floor(rat_y * sze_y)
    nmh_y = n_mid_y // 2
    n_mid_x = math.floor(rat_x * sze_x)
    nmh_x = n_mid_x // 2
    cnt_z, cnt_y, cnt_x = (
        out.shape[0] // 2,
        out.shape[1] // 2,
        out.shape[2] // 2,
    )
    return out[
        cnt_z - nmh_z : cnt_z + nmh_z,
        cnt_y - nmh_y : cnt_y + nmh_y,
        cnt_x - nmh_x : cnt_x + nmh_x,
    ]
