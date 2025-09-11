"""Richardson Lucy deconvolution"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.tree_util import Partial
from jaxtyping import Array, Num

from ..gauss import sampled_gaussian_kernel_1d

__all__ = ["deconvolve", "deconvolve_gaussian", "deconvolve_separable"]


@partial(jax.jit, static_argnums=(3, 4))
def deconvolve_separable(
    x: Array,
    psfs: Num[Array, "n_dim a"],
    bps: Num[Array, "n_dim b"],
    n_dim: int,
    num_iter: int,
) -> Array:
    psf_list = [psf.flatten() for psf in jnp.split(psfs, n_dim, 0)]
    bps_list = [bp.flatten() for bp in jnp.split(bps, n_dim, 0)]
    if n_dim == 2:
        return _decon_sep2(x, *psf_list, *bps_list, num_iter)
    else:
        return _decon_sep3(x, *psf_list, *bps_list, num_iter)


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
        3,
    ),
)
def deconvolve_gaussian(
    x: Array,
    sigma_lat: float,
    sigma_ax: float | None,
    num_iter: int,
) -> Array:
    """Deconvolve the input array using a Gaussian PSF with specified standard deviations in the lateral, axial dimensions.

    Args:
        x (Array): input image or volume (2 or 3D)
        sigma_lat (float): lateral standard deviation
        sigma_ax (float): axial standard deviation (only used if input is 3D)
        num_iter (int): number of iterations

    Returns:
        Array: deconvolved array
    """
    n_dim = 2 if sigma_ax is None else 3

    psf_lat = sampled_gaussian_kernel_1d(sigma_lat, 1e-8, n_dim)
    bp_lat = psf_lat[::-1].copy()
    if sigma_ax is None:
        return _decon_sep2(x, psf_lat, psf_lat, bp_lat, bp_lat, num_iter)
    else:
        psf_ax = sampled_gaussian_kernel_1d(sigma_ax, 1e-8, n_dim)
        bp_ax = psf_ax[::-1].copy()
        return _decon_sep3(
            x, psf_ax, psf_lat, psf_lat, bp_ax, bp_lat, bp_lat, num_iter
        )


def deconvolve(
    x: Array,
    psf: Array,
    backproj: Array,
    num_iter: int,
) -> Array:
    """Deconvolve the input array by Richarson-Lucy deconvolution.

    Args:
        x (Array): input array to be deconvolved.
        psf (Array): point spread function.
        backproj (Array): backprojection array.
        num_iter (int): number of iterations

    Returns:
        Array
    """

    def _single_iter(curr: Array, _) -> tuple[Array, None]:
        return _decon_single_iter(x, curr, psf, backproj), None

    dec, _ = jax.lax.scan(_single_iter, x, None, length=num_iter)
    return dec


@partial(jax.jit, static_argnums=(5,))
def _decon_sep2(
    x: Array,
    psf_y: Num[Array, " a"],
    psf_x: Num[Array, " b"],
    bp_y: Num[Array, " a"],
    bp_x: Num[Array, " b"],
    num_iter: int,
) -> Array:
    def _single_iter(curr: Array, _) -> tuple[Array, None]:
        return (
            _decon_sep2_single_iter(x, curr, psf_y, bp_y, psf_x, bp_x),
            None,
        )

    dec, _ = jax.lax.scan(_single_iter, x, None, length=num_iter)
    return dec


@partial(jax.jit, static_argnums=(7,))
def _decon_sep3(
    x: Array,
    psf_z: Num[Array, " a"],
    psf_y: Num[Array, " b"],
    psf_x: Num[Array, " c"],
    bp_z: Num[Array, " a"],
    bp_y: Num[Array, " b"],
    bp_x: Num[Array, " c"],
    num_iter: int,
) -> Array:
    """Deconvolve the input array by Richardson-Lucy deconvolution using separable filters along each axis.

    Args:
        x (Array): input array to be deconvolved.
        psfs (Sequence[Num[Array, " a"]]): sequence of PSF kernels, one per axis.
        backprojs (Sequence[Num[Array, " b"]]): sequence of backprojector kernels, one per axis.
        num_iter (int): number of iterations.

    Raises:
        ValueError: only works for 2- and 3-D arrays.

    Returns:
        Array
    """

    def _single_iter(curr: Array, _) -> tuple[Array, None]:
        return (
            _decon_sep3_single_iter(
                x,
                curr,
                psf_z,
                bp_z,
                psf_y,
                bp_y,
                psf_x,
                bp_x,
            ),
            None,
        )

    dec, _ = jax.lax.scan(_single_iter, x, None, length=num_iter)
    return dec


def deconvolve_dualview(
    a: Array,
    b: Array,
    psf_a: Array,
    psf_b: Array,
    bp_a: Array,
    bp_b: Array,
    num_iter: int,
    epsilon: float = 1e-5,
) -> Array:
    """Jointly deconvolve the two input arrays, a and b, using the additive Richardson Lucy algorithm.

    Args:
        a (Array): first array
        b (Array): second array
        psf_a (Array): point spread function for first array
        psf_b (Array): point spread function for second array
        bp_a (Array): backprojector for first array
        bp_b (Array): backprojector for second array
        num_iter (int): number of iterations
        epsilon (float, optional): small parameter to prevent division by 0. Defaults to 1e-5.

    Returns:
        Array: deconvolved array
    """

    @partial(jax.jit, static_argnums=(8,))
    def _single_iter(
        a: Array,
        b: Array,
        curr: Array,
        _,
        psf_a: Array,
        psf_b: Array,
        bp_a: Array,
        bp_b: Array,
        eps: float,
    ) -> tuple[Array, None]:
        con_a = convolve(curr, psf_a, mode="same")
        est_a = jnp.multiply(
            curr,
            convolve(
                jnp.where(con_a > eps, jnp.divide(a, con_a), 0.0),
                bp_a,
                mode="same",
            ),
        )
        con_b = convolve(curr, psf_b, mode="same")
        est_b = jnp.multiply(
            curr,
            convolve(
                jnp.where(con_b > eps, jnp.divide(b, con_b), 0.0),
                bp_b,
                mode="same",
            ),
        )
        return jnp.divide(jnp.add(est_a, est_b), 2), None

    single_iter = Partial(
        _single_iter,
        a,
        b,
        psf_a=psf_a,
        psf_b=psf_b,
        bp_a=bp_a,
        bp_b=bp_b,
        eps=epsilon,
    )

    init = jnp.divide(jnp.add(a, b), 2)
    dec, _ = jax.lax.scan(single_iter, init, None, length=num_iter)
    return dec


def _decon_single_iter(
    data: Array, curr: Array, psf: Array, backproj: Array
) -> Array:
    return jnp.multiply(
        curr,
        convolve(
            jnp.divide(data, convolve(curr, psf, mode="same")),
            backproj,
            mode="same",
        ),
    )


def _conv2d_separable(
    x: Num[Array, "y x"],
    hrow: Num[Array, " m"],
    hcol: Num[Array, " n"],
) -> Num[Array, "y x"]:
    return jnp.apply_along_axis(
        Partial(jnp.convolve, v=hcol, mode="same"),
        1,
        jnp.apply_along_axis(Partial(jnp.convolve, v=hrow, mode="same"), 0, x),
    )


def _conv3d_separable(
    x: Num[Array, "z y x"],
    hdep: Num[Array, " k"],
    hrow: Num[Array, " m"],
    hcol: Num[Array, " n"],
) -> Num[Array, "z y x"]:
    return jnp.apply_along_axis(
        Partial(jnp.convolve, v=hdep, mode="same"),
        0,
        jnp.apply_along_axis(
            Partial(jnp.convolve, v=hrow, mode="same"),
            1,
            jnp.apply_along_axis(
                Partial(jnp.convolve, v=hcol, mode="same"), 2, x
            ),
        ),
    )


def _decon_sep2_single_iter(
    data: Num[Array, "y x"],
    curr: Num[Array, "y x"],
    psf_r: Num[Array, " m"],
    backproj_r: Num[Array, " m"],
    psf_c: Num[Array, " n"],
    backproj_c: Num[Array, " n"],
) -> Num[Array, "y x"]:
    return jnp.multiply(
        curr,
        _conv2d_separable(
            jnp.divide(data, _conv2d_separable(curr, psf_r, psf_c)),
            backproj_r,
            backproj_c,
        ),
    )


def _decon_sep3_single_iter(
    data: Num[Array, "y x"],
    curr: Num[Array, "y x"],
    psf_z: Num[Array, " k"],
    backproj_z: Num[Array, " k"],
    psf_r: Num[Array, " m"],
    backproj_r: Num[Array, " m"],
    psf_c: Num[Array, " n"],
    backproj_c: Num[Array, " n"],
) -> Num[Array, "z y x"]:
    return jnp.multiply(
        curr,
        _conv3d_separable(
            jnp.divide(data, _conv3d_separable(curr, psf_z, psf_r, psf_c)),
            backproj_z,
            backproj_r,
            backproj_c,
        ),
    )
