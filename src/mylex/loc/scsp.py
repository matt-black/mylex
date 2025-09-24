"""Scale Spaces and Associated Functions/Utilities

Built largely off of the `pyscscp` library of Tony Lindeberg (https://github.com/tonylindeberg/pyscscp), but adapted for use with JAX.
"""

import math
from collections.abc import Sequence
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.scipy.special import erf
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Float, Int
from scipy.special import erfcinv, ive

__all__ = [
    "detect_peaks_at_size",
    "multiscale_peak_mask",
    "conv_gauss",
    "diff_gauss",
    "njet",
    "discrete_gaussian_kernel_1d",
    "sampled_gaussian_kernel_1d",
    "integrated_gaussian_kernel_1d",
    "sampled_gaussian_derivative_kernel_1d",
    "integrated_gaussian_derivative_kernel_1d",
    "gaussian",
    "scaled_erf",
    "truncate_filter",
    "mirror_half_filter",
    "gaussian_filter_size",
    "gaussian_derivative_filter_size",
    "truncation_error_nd_to_1d",
    "discrete_delta",
    "scale_norm_factor",
    "discrete_derivative_mask",
    "local_maxima_3d",
    "local_maxima_4d",
]


def multiscale_detection_scalespace(
    x: Float[Array, "z y x"] | Float[Array, "y x"],
    scales: tuple[float, ...],
    sigma_lat: float,
    sigma_ax: float | None = None,
) -> Float[Array, "s z y x"] | Float[Array, "s y x"]:
    """Construct a scalespace for multi-scale peak detection of fluorescent (assumed bright) objects.

    Args:
        x (Float[Array, "z y x"] | Float[Array, "y x"]): input image or volume
        scales (tuple[float,...]): scales to probe, relative to the specified sigmas.
        sigma_lat (float): lateral resolution, standard deviation.
        sigma_ax (float): axial resolution, standard deviation.

    Returns:
        Float[Array, "s z y x"]|Float[Array, "s y x"]: scale space
    """
    if sigma_ax is None:

        def nj_fun(scl: float) -> Float[Array, "y x"]:
            return njet(
                x,
                "sqrtdetHess",
                (sigma_lat * scl, sigma_lat * scl),
                method="sampled",
                normalize=False,
                epsilon=1e-2,
            )

    else:

        def nj_fun(scl: float) -> Float[Array, "y x"]:
            return njet(
                x,
                "sqrtdetHess",
                (sigma_ax * scl, sigma_lat * scl, sigma_lat * scl),
                method="sampled",
                normalize=False,
                epsilon=1e-2,
            )

    return jnp.negative(jnp.stack([nj_fun(scale) for scale in scales], axis=0))


@partial(jax.jit, static_argnums=(1, 2, 3))
def multiscale_peak_mask(
    x: Float[Array, "z y x"] | Float[Array, "y x"],
    scales: tuple[float, ...],
    sigma_lat: float,
    sigma_ax: float | None = None,
) -> Int[Array, "z y x"] | Int[Array, "y x"]:
    """Detect peaks at multiple scales by searching for local maxima in scale space.

    Args:
        x (Float[Array, "z y x"] | Float[Array, "y x"]): input image or volume
        scales (tuple[float,...]): scales to probe, relative to the specified sigmas.
        sigma_lat (float): lateral resolution, standard deviation.
        sigma_ax (float): axial resolution, standard deviation.

    Returns:
        Int[Array, "z y x"]|Int[Array, "y x"]: mask where nonzero values are peaks and values correspond to the detection scale

    Notes:
        The output type is a 4 bit unsigned integer (uint4) and thus using more than 15 detection scales (length of `scales` tuple is 17) will cause overflow problems.
    """
    scsp = multiscale_detection_scalespace(x, scales, sigma_lat, sigma_ax)
    if sigma_ax is None:
        mask_fun = _local_maxima_3d_mask
        padding = ((1, 1), (1, 1))
    else:
        mask_fun = _local_maxima_4d_mask
        padding = ((1, 1), (1, 1), (1, 1))
    mask = jnp.zeros_like(scsp, shape=scsp.shape[1:], dtype=jnp.uint4)

    def body_fun(
        i: int, msk: Int[Array, "z y x"] | Int[Array, "y x"]
    ) -> Int[Array, "z y x"] | Int[Array, "y x"]:
        new_mask = (
            jnp.pad(
                mask_fun(jax.lax.dynamic_slice_in_dim(scsp, i - 1, 3, 0)),
                padding,
                mode="constant",
                constant_values=0,
            ).astype(jnp.uint4)
            * i
        )
        return jnp.add(msk, new_mask)

    return jax.lax.fori_loop(1, len(scales) - 1, body_fun, mask)


def detect_peaks_at_size(
    x: Float[Array, "z y x"] | Float[Array, "y x"],
    sigma_lat: float,
    sigma_ax: float | None = None,
) -> Int[Array, "n 3"] | Int[Array, "n 2"]:
    """Detect peaks (Gaussian blobs) of the specified width in an array.

    This method constructs a dyadic scale space around the specified size, then looks for peaks in that scale space at the scale corresponding to the specified `sigmas`. Peaks are found by looking for local maxima in the scale space.

    Args:
        x (Float[Array, "z y x"]|Float[Array, "y x"]): array to detect peaks in
        sigma_lat (float): lateral resolution as standard deviation
        sigma_ax (float|None): axial resolution as standard deviation

    Returns:
        Int[Array, "n 3"]|Int[Array, "n 2"]: coordinates of the detected peaks.
    """
    if sigma_ax is None:
        return _detect_peaks_at_size_2d(x, sigma_lat)
    elif x.ndim == 3:
        return _detect_peaks_at_size_3d(x, sigma_ax, sigma_lat)
    else:
        raise ValueError("only works for 2D or 3D inputs")


def _detect_peaks_at_size_3d(
    x: Float[Array, "z y x"],
    sigma_ax: float,
    sigma_lat: float,
) -> Int[Array, "n 3"]:
    # generate the scale space
    scsp = multiscale_detection_scalespace(
        x, (0.5, 1.0, 2.0), sigma_lat, sigma_ax
    )
    # detect peaks
    pks = local_maxima_4d(scsp)
    # trim out the first column, which is just 1's
    return pks[:, 1:]


def _detect_peaks_at_size_2d(
    x: Float[Array, "y x"],
    sigma: float,
) -> Int[Array, "n 2"]:
    scsp = multiscale_detection_scalespace(x, (0.5, 1.0, 2.0), sigma, None)
    # trim out the edges where zero-padding was done, "invalid" convolutions
    pks = local_maxima_3d(scsp)
    return pks[:, 1:]


@partial(jax.jit, static_argnums=(1, 2, 3))
def conv_gauss(
    x: Float[Array, "y x"] | Float[Array, "z y x"],
    sigma: tuple[float | int, ...],
    epsilon: float = 1e-8,
    method: str = "sampled",
) -> Float[Array, "y x"] | Float[Array, "z y x"]:
    """Convolve the input with a Gaussian.

    Args:
        x (Float[Array, "y x"] | Float[Array, "z y x"]): input array, must be <= 2 or 3-dimensional.
        sigma (tuple[float | int, ...]): standard deviation of Gaussians.
        epsilon (float, optional): relative truncation error. Defaults to 1e-8.
        method (str, optional): Method for generating the Gaussian kernel; one of ('sampled', 'discrete'). Defaults to 'sampled'.

    Returns:
        Float[Array, "y x"]|Float[Array, "z y x"]
    """
    n_dim = len(x.shape)
    if method == "discrete":
        filter_fun = discrete_gaussian_kernel_1d
    elif method == "sampled":
        filter_fun = sampled_gaussian_kernel_1d
    else:
        raise ValueError("invalid method")
    filts = [filter_fun(s, epsilon, n_dim) for s in sigma]
    sizes = [f.shape[0] for f in filts]
    if n_dim == 2:
        padding = tuple([(s - 1 - (s - 1) // 2, (s - 1) // 2) for s in sizes])
    else:
        padding = tuple([(s // 2, s - s // 2 - 1) for s in sizes])
    x = jnp.pad(x, padding, mode="symmetric")

    if n_dim == 1:
        return jnp.convolve(x, filts[0], mode="valid")
    elif n_dim == 2:
        kern = jnp.outer(filts[0], filts[1])
        return convolve(x, kern, mode="valid")
    else:
        for axis, filt in enumerate(filts):
            x = jnp.apply_along_axis(
                Partial(jnp.convolve, v=filt, mode="valid"), axis, x
            )
        return x


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
        3,
        4,
        5,
        6,
    ),
)
def diff_gauss(
    x: Float[Array, "y x"] | Float[Array, "z y x"],
    sigma: tuple[int | float, ...],
    order: tuple[int, ...],
    epsilon: float = 1e-4,
    normalize: bool = False,
    gamma: float = 1.0,
    method: str = "sampled",
) -> Float[Array, "y x"] | Float[Array, "z y x"]:
    """Differentiate, to specified order, the input at the specified scale by convolution with the derivative of a Gaussian.

    Args:
        x (Float[Array, "y x"] | Float[Array, "z y x"]): input array.
        sigma (tuple[int | float, ...]): standard deviation along each axis.
        order (tuple[int, ...]): derivative order to compute along each axis.
        epsilon (float, optional): relative truncation error. Defaults to 1e-8.
        normalize (bool, optional): normalize the output. Defaults to False.
        gamma (float, optional): scale normalization factor, only used if `normalize=True`. Defaults to 1.0.
        method (str, optional): method for generating the differentiation kernel. One of 'sampled' or 'integrated'. Defaults to 'sampled'.

    Returns:
        Float[Array, "y x"] | Float[Array, "z y x"]
    """
    n_dim = x.ndim
    # compute size of the kernel to use
    # NOTE: to ensure that the output size is not increased by the size of the convolution filter, we limit it to be at most as big as the input image. For very small images (or very large sigma), this may lead to unexpectedly high truncation errors because this will override epsilon. In preliminary testing, doesn't seem to matter much.
    size = [
        min(gaussian_derivative_filter_size(s, o, epsilon, n_dim), dim_shp)
        for s, o, dim_shp in zip(sigma, order, x.shape)
    ]
    if normalize:
        norm_factor = reduce(
            lambda x, y: x + y, [s ** (o * gamma) for s, o in zip(sigma, order)]
        )
    else:
        norm_factor = 1.0
    filt_fun = (
        sampled_gaussian_derivative_kernel_1d
        if method == "sampled"
        else integrated_gaussian_derivative_kernel_1d
    )

    filts = [
        filt_fun(std, ord, sze) for std, ord, sze in zip(sigma, order, size)
    ]
    # NOTE: actual size taken by filter functions is 2*size+1, so we need to account for that when padding the array.
    # NOTE: do padding here because we want to use symmetric padding to minimize boundary artifacts, whereas convolution functions use zero-padding if mode="same" is used.
    size_full = [2 * s + 1 for s in size]
    if n_dim == 2:
        padding = tuple(
            [(s - 1 - (s - 1) // 2, (s - 1) // 2) for s in size_full]
        )
    else:
        padding = tuple([(s // 2, s - s // 2 - 1) for s in size_full])
    x = jnp.pad(x, padding, mode="symmetric")

    # switch on dimensionality here is based on empirical benchmarking that when the number of dimensions exceeds 3, it is faster to do `apply_along_axis` along each axis than it is to do the full ND convolution. note that we can do this because convolution with an n-dimensional Gaussian is separable.
    if n_dim == 1:
        return jnp.multiply(
            norm_factor, jnp.convolve(x, filts[0], mode="valid")
        )
    elif n_dim == 2:
        kern = jnp.outer(filts[0], filts[1])
        return jnp.multiply(norm_factor, convolve(x, kern, mode="valid"))
    else:
        for axis, filt in enumerate(filts):
            x = jnp.apply_along_axis(
                Partial(jnp.convolve, v=filt, mode="valid"),
                axis,
                x,
            )
        return jnp.multiply(x, norm_factor)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6))
def njet(
    x: Float[Array, "y x"] | Float[Array, "z y x"],
    function: str,
    sigma: tuple[int | float, ...],
    method: str = "sampled",
    normalize: bool = True,
    gamma: float = 1.0,
    epsilon: float = 1e-4,
) -> Float[Array, "y x"] | Float[Array, "z y x"]:
    """Compute an N-jet function in terms of scale-normalized Gaussian derivatives of the input image or volume. The scale level is set by `sigma` and may be specified per-axis.

    Args:
        x (Float[Array, "y x"] | Float[Array, "z y x"]): input array to compute N-jet function of.
        function (str): The N-jet function to compute. See "Implemented Functions" for a list of N-jet functions that can be computed.
        sigma (tuple[int | float, ...]): standard deviation to use for the Gaussian convolution along each axis.
        method (str): method to generate the Gaussian kernel.
        normalize (bool): whether to normalize the response using scale normalization.
        gamma (float, optional): scale normalization factor. Defaults to 1.0.
        epsilon (float, optional): relative truncation error. Defaults to 1e-4.

    Returns:
        Float[Array, "y x"]|Float[Array, "z y x"]

    Implemented Functions:
        'L'           : smoothed scale space representation
        'Lx'          : 1st order derivative in x-direction
        'Ly'          : 1st order derivative in y-direction
        'Lz'          : 1st order derivative in z-direction
        'Lxx'         : 2nd order derivative in x-direction
        'Lyy'         : 2nd order derivative in y-direction
        'Lzz'         : 2nd order derivative in z-direction
        'Lxy'         : mixed 2nd-order derivative in x- and y-directions.
        'Lxz'         : mixed 2nd-order derivative in x- and z-directions.
        'Lyz'         : mixed 2nd-order derivative in y- and z-directions.
        'Lv'          : gradient magnitude
        'Lv2'         : squared gradient magnitude
        'Laplace'     : laplacian
        'detHess'     : determinant of the hessian
        'sqrtdetHess' : signed square root of abs. val. determinant of hessian
    """
    if x.ndim > 3 or x.ndim < 2:
        raise ValueError("input must be 2- or 3-dimensional")
    two_d = x.ndim == 2
    njet_fun = Partial(
        njet,
        x,
        sigma=sigma,
        method=method,
        normalize=normalize,
        gamma=gamma,
        epsilon=epsilon,
    )
    function = function.upper()  # remove case-sensitivity
    if function == "L":
        return conv_gauss(x, sigma, epsilon, method)
    elif function[0] == "L":
        if function == "LAPLACE":
            # explicit case for laplace
            dxx = njet_fun("Lxx")
            dyy = njet_fun("Lyy")
            if two_d:
                return jnp.add(dxx, dyy)
            else:
                dzz = njet_fun("Lzz")
                return jnp.add(jnp.add(dxx, dyy), dzz)
        elif "V" not in function:
            # this case covers all Lx, Lxx, Lxy, Lxyz combinations and is the main workhorse of this function. note that a lot of other `function` options end up doing their computation here.
            ox = sum(list(map(lambda c: c == "X", function[1:])))
            oy = sum(list(map(lambda c: c == "Y", function[1:])))
            if two_d:
                return diff_gauss(x, sigma, (oy, ox), epsilon, normalize, gamma)
            else:
                oz = sum(list(map(lambda c: c == "Z", function[1:])))
                return diff_gauss(
                    x, sigma, (oz, oy, ox), epsilon, normalize, gamma
                )
        else:
            # this case covers 'Lv' or 'Lv2'
            dx = njet_fun("Lx")
            dy = njet_fun("Ly")
            if function == "Lv":
                if two_d:
                    return jnp.sqrt(jnp.add(jnp.square(dx), jnp.square(dy)))
                else:
                    dz = njet_fun("Lz")
                    return jnp.sqrt(
                        jnp.add(
                            jnp.add(jnp.square(dx), jnp.square(dy)),
                            jnp.square(dz),
                        )
                    )
            elif function == "Lv2":
                if two_d:
                    return jnp.add(jnp.square(dx), jnp.square(dy))
                else:
                    dz = njet_fun("Lz")
                    return jnp.add(
                        jnp.add(jnp.square(dx), jnp.square(dy)), jnp.square(dz)
                    )
            else:
                raise ValueError("invalid njet function")
    elif "HESS" in function:  # we're computing a hessian of some sort
        dxx = njet_fun("Lxx")
        dyy = njet_fun("Lyy")
        dxy = njet_fun("Lxy")
        if two_d:
            det_hess = jnp.multiply(dxx, dyy) - jnp.square(dxy)
            if function.startswith("DET"):
                return det_hess
            elif function.startswith("SQRTDET"):
                return jnp.sign(det_hess) * jnp.sqrt(jnp.abs(det_hess))
            else:
                raise ValueError("invalid njet function")
        else:  # there-dimensional determinant of hessian
            dzz = njet_fun("Lzz")
            dxz = njet_fun("Lxz")
            dyz = njet_fun("Lyz")
            t1 = jnp.multiply(
                dxx, jnp.subtract(jnp.multiply(dyy, dzz), jnp.square(dyz))
            )
            t2 = jnp.negative(
                jnp.multiply(
                    dxy,
                    jnp.subtract(
                        jnp.multiply(dxy, dzz), jnp.multiply(dyz, dxz)
                    ),
                )
            )
            t3 = jnp.multiply(
                dxz,
                jnp.subtract(jnp.multiply(dxy, dyz), jnp.multiply(dyy, dxz)),
            )
            det_hess = jnp.add(jnp.add(t1, t2), t3)
            if function.startswith("DET"):
                return det_hess
            elif function.startswith("SQRTDET"):
                return jnp.sign(det_hess) * jnp.sqrt(jnp.abs(det_hess))
            else:
                raise ValueError("invalid njet function")
    else:
        raise ValueError("invalid njet function")


def discrete_gaussian_kernel_1d(
    sigma: float,
    epsilon: float = 1e-8,
    dim: int = 1,
) -> Float[Array, " x"]:
    """Generate 1D discrete analogue of the Gaussian kernel with specified standard deviation and relative truncation error.

    Args:
        sigma (float): standard deviation of Gaussian.
        epsilon (float, optional): relative truncation error. Defaults to 1e-8.
        dim (int, optional): dimensionality of signal. Defaults to 1.

    Returns:
        Float[Array, " x"]
    """
    var = sigma**2
    size = math.ceil(1 + 1.5 * gaussian_filter_size(sigma, epsilon, dim))
    if sigma == 0:
        return discrete_delta(size)
    else:
        # generate filter coeff. from modified Bessel functions
        # FIXME: the use of scipy.special.ive here prevents Jax JIT compilation
        longhalf_filtvec = ive(jnp.arange(0, size), var)
        half_filtvec = truncate_filter(
            longhalf_filtvec, truncation_error_nd_to_1d(epsilon, dim)
        )
        return mirror_half_filter(half_filtvec)


def sampled_gaussian_kernel_1d(
    sigma: float,
    epsilon: float = 1e-8,
    dim: int = 1,
) -> Float[Array, " x"]:
    """Generate a 1D Gaussian filter by sampling from the Gaussian function at discrete spatial coordinates.

    Args:
        sigma (float): standard deviation of Gaussian
        epsilon (float, optional): relative truncation error. Defaults to 1e-8.
        dim (int, optional): dimensionality of signal. Defaults to 1.

    Returns:
        Float[Array, " x"]
    """
    size = math.ceil(1.1 * gaussian_filter_size(sigma, epsilon, dim))
    if sigma == 0:
        return discrete_delta(size)
    else:
        x = jnp.linspace(-size, size, 1 + 2 * size)
        return gaussian(x, sigma)


def integrated_gaussian_kernel_1d(
    sigma: float,
    epsilon: float = 1e-8,
    dim: int = 1,
) -> Float[Array, " x"]:
    """Generate a 1D box-integrated Gaussian kernel with specified standard deviation.

    Args:
        sigma (float): standard deviation.
        epsilon (float, optional): relative truncation error. Defaults to 1e-8.
        dim (int, optional): dimensionality of signal. Defaults to 1.

    Returns:
        Float[Array, " x"]

    Notes:
        At coarser scales, the kernel has additional spatial variance (1/12).
    """
    size = math.ceil(1.1 * gaussian_filter_size(sigma, epsilon, dim))
    x = jnp.linspace(-size, size, 1 + 2 * size)
    return scaled_erf(x + 0.5, sigma) - scaled_erf(x - 0.5, sigma)


def sampled_gaussian_derivative_kernel_1d(
    sigma: float,
    order: int,
    size_or_x: int | Float[Array, " x"],
) -> Float[Array, " x"]:
    """Generate a 1D derivative of Gaussian kernel of specified order by sampling the derivative of the Gaussian at discrete spatial coordinates.

    Args:
        sigma (float): standard deviation.
        order (int): order of the derivative.
        size_or_x (int | Float[Array, " x"]): either the size of the filter, or a spatial coordinate to evaluate the derivative at.

    Returns:
        Float[Array, " x"]:
    """
    if sigma == 0:
        if isinstance(size_or_x, int):
            return discrete_delta(size_or_x)
        else:
            return discrete_delta(size_or_x.shape[0] // 2)
    else:
        if isinstance(size_or_x, int):
            x = jnp.linspace(-size_or_x, size_or_x, 1 + 2 * size_or_x)
        else:
            x = size_or_x
        var = sigma**2
        g = jnp.exp(-jnp.square(x) / (2 * var))
        if order == 0:
            return g
        elif order == 1:
            return (-x / var) * g
        elif order == 2:
            return ((jnp.square(x) - var) / jnp.square(var)) * g
        elif order == 3:
            return (-(jnp.power(x, 3) - 3 * var * x) / jnp.power(var, 3)) * g
        elif order == 4:
            return (
                jnp.power(x, 4)
                - 6 * var * jnp.square(x)
                + 3 * jnp.square(var) / jnp.power(var, 4)
            ) * g
        else:
            raise ValueError(
                "Gaussian derivaties only implemented up to order 4."
            )


def integrated_gaussian_derivative_kernel_1d(
    sigma: float,
    order: int,
    size: int,
) -> Float[Array, " x"]:
    """Make an integrated Gaussian derivative kernel of specified order and standard deviation sigma.

    Args:
        sigma (float): standard deviation.
        order (int): order of the derivative.
        size (int): size of the filter.

    Returns:
        Float[Array, " x"]
    """
    if sigma == 0:
        return discrete_delta(size)
    else:
        x = jnp.linspace(-size, size, 1 + 2 * size)
        if order == 0:
            return scaled_erf(x + 0.5, sigma) - scaled_erf(x - 0.5, sigma)
        else:
            kernel_fun = partial(
                sampled_gaussian_derivative_kernel_1d, sigma, order - 1
            )
            return kernel_fun(x + 0.5) - kernel_fun(x - 0.5)


def gaussian(x: Float[Array, " x"], sigma: float) -> Float[Array, " x"]:
    """Compute values of the Gaussian function at 1d coordinates, `x`.

    Args:
        x (Float[Array, " x"]): spatial coordinate vector
        sigma (float): standard deviation of the Gaussian

    Returns:
        Float[Array, " x"]
    """
    var = sigma**2
    return jnp.exp(-(jnp.square(x) / (2 * var))) / (
        jnp.sqrt(2 * math.pi) * sigma
    )


def scaled_erf(x: Float[Array, " x"], sigma: float) -> Float[Array, " x"]:
    """Compute scaled error function.

    Args:
        x (Float[Array, " x"]): coordinates to compute scaled error function at.
        sigma (float): standard deviation.

    Returns:
        Float[Array, " x"]
    """
    return 0.5 * (1 + erf(x / jnp.sqrt(2) * sigma))


def truncate_filter(
    half_filtvec_overest: Float[Array, " x"],
    epsilon: float,
) -> Float[Array, " x"]:
    """truncate_filter Truncate the input filter with overestimated size to a more compact size based on the tolerable error, `epsilon`.

    Args:
        half_filtvec_overest (Float[Array, " x"]): overestimated initial filter, to be trimmed.
        epsilon (float): tolerable error.

    Returns:
        Float[Array, " x"]: truncated filter
    """
    n = half_filtvec_overest.shape[0]
    filt_sum = half_filtvec_overest[0]
    i = 1
    while (filt_sum < 1 - epsilon) and (i < n):
        filt_sum = filt_sum + 2 * half_filtvec_overest[i]
        i += 1
    return half_filtvec_overest[:i]


def mirror_half_filter(half_filter: Float[Array, " x"]) -> Float[Array, " x"]:
    """Extend a one-sided spatial filter to a symmetric filter by spatial mirroring.

    Args:
        half_filter (Float[Array, " x"]): one-sided filter to be symmetrized.

    Returns:
        Float[Array, " x"]
    """
    n = half_filter.shape[0]
    rev = half_filter[::-1]
    return jnp.append(rev[: n - 1], half_filter)


def gaussian_filter_size(
    sigma: float,
    epsilon: float,
    dim: int,
) -> float:
    """Compute length of vector to truncate Gaussian kernel with such that there is a relative truncation error `epsilon` over a `dim`-dimensional domain.

    Args:
        sigma (float): standard deviation of Gaussian.
        epsilon (float): relative truncation error.
        dim (int): number of dimensions in domain.

    Returns:
        float
    """
    var = sigma * sigma
    eps_1d = truncation_error_nd_to_1d(epsilon, dim)
    n = math.sqrt(2 * var) * erfcinv(eps_1d)
    return n


def gaussian_derivative_filter_size(
    sigma: float, order: int, epsilon: float, dim: int
) -> int:
    """Estimate the minimum size for truncating a Gaussian derivative kernel such that the relative truncation error for a 1D derivative convolution is less than `epsilon`.

    Args:
        sigma (float): standard deviation of Gaussian.
        order (int): order of derivative to compute (>=0).
        epsilon (float): relative truncation error.
        dim (int): dimensionality of signal.

    Returns:
        int
    """
    if order == 0:
        return math.ceil(gaussian_filter_size(sigma, epsilon, dim))
    else:
        eps_1d = truncation_error_nd_to_1d(epsilon, dim)
        dsize = gaussian_filter_size(sigma, eps_1d / 2**order, 1)
        return 1 + order + math.ceil(dsize)


def truncation_error_nd_to_1d(epsilon: float, dim: int) -> float:
    """Convert relative truncation error over `dim`-dimensional domain to a relative truncation error over a 1D domain.

    Args:
        epsilon (float): N-dimensional relative truncation error
        dim (int): number of dimensions (N)

    Returns:
        float
    """
    return 1 - (1 - epsilon) ** (1 / dim)


def discrete_delta(
    *sizes: int,
) -> Float[Array, " x"] | Float[Array, "y x"] | Float[Array, "z y x"]:
    """Generate a discrete delta function.

    Args:
        *sizes (int): per-axis size of output.

    Returns:
        Float[Array, " x"]|Float[Array, "y x"]|Float[Array, "z y x"]
    """
    if len(sizes) == 1:
        out = jnp.zeros(sizes)
        if sizes[0] % 2:
            c = (sizes[0] - 1) // 2
        else:
            c = sizes[0] // 2
        out = out.at[c].set(1.0)
    elif len(sizes) == 2:
        nrow, ncol = sizes
        out = jnp.zeros((nrow, ncol))
        yc = (nrow - 1) // 2 if nrow % 2 else nrow // 2
        xc = (ncol - 1) // 2 if ncol % 2 else ncol // 2
        out = out.at[yc, xc].set(1.0)
    elif len(sizes) == 3:
        ndep, nrow, ncol = sizes
        out = jnp.zeros((ndep, nrow, ncol))
        zc = (ndep - 1) // 2 if ndep % 2 else ndep // 2
        yc = (nrow - 1) // 2 if nrow % 2 else nrow // 2
        xc = (ncol - 1) // 2 if ncol % 2 else ncol // 2
        out = out.at[zc, yc, xc].set(1.0)
    else:
        raise ValueError(
            f"Number of specified sizes must be 1, 2, or 3. Got {len(sizes)}."
        )
    return out


def scale_norm_factor(
    sigmas: Sequence[int | float],
    orders: Sequence[int],
    normalize: bool = True,
    gamma: float = 1.0,
) -> float:
    """Compute the scale normalization factor for scale-normalized Gaussian derivative with per-axis standard deviation `sigmas` and of orders `orders`.

    Args:
        sigmas (Sequence[int | float]): per-axis standard deviation.
        orders (Sequence[int]): per-axis order of differentiation.
        normalize (bool, optional): whether to do normalization or not. Defaults to True.
        gamma (float, optional): scale normalization factor. Defaults to 1.0.

    Returns:
        float
    """
    if normalize:
        return reduce(
            lambda x, y: x + y,
            [s ** (gamma * o) for s, o in zip(sigmas, orders)],
        )
    else:
        return 1.0


def discrete_derivative_mask(
    axes: str,
    out_dim: int,
) -> Float[Array, " x"] | Float[Array, "y x"] | Float[Array, "z y x"]:
    if out_dim > 3 or out_dim < 1:
        raise ValueError("out_dim must be in (1,2,3)")
    if axes == "x":
        msk = jnp.array([-0.5, 0.0, 0.5])
        if out_dim == 1:
            return msk
        elif out_dim == 2:
            return msk[None, :]
        else:
            return msk[None, None, :]
    elif axes == "y":
        msk = jnp.array([0.5, 0.0, -0.5])
        if out_dim == 1:
            return msk
        elif out_dim == 2:
            return msk[:, None]
        else:
            return msk[None, :, None]
    elif axes == "z":
        msk = jnp.array([0.5, 0.0, -0.5])
        if out_dim == 1:
            return msk
        elif out_dim == 2:
            return msk[:, None]
        else:
            return msk[:, None, None]
    elif axes == "xx":
        msk = jnp.array([1.0, -2.0, 1.0])
        if out_dim == 1:
            return msk
        elif out_dim == 2:
            return msk[None, :]
        else:
            return msk[None, None, :]
    elif axes == "yy":
        msk = jnp.array([1.0, -2.0, 1.0])
        if out_dim == 1:
            return msk
        elif out_dim == 2:
            return msk[:, None]
        else:
            return msk[None, :, None]
    elif axes == "zz":
        msk = jnp.array([1.0, -2.0, 1.0])
        if out_dim == 1:
            return msk
        elif out_dim == 2:
            return msk[:, None]
        else:
            return msk[:, None, None]
    elif axes == "xy" or axes == "yz" or axes == "xz":
        msk = jnp.array([[-0.25, 0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, -0.25]])
        if out_dim == 1:
            raise ValueError("can't take a 2D derivative for 1D output")
        if axes == "xy":
            return msk if out_dim == 2 else msk[None, :, :]
        elif axes == "yz":
            return msk if out_dim == 2 else msk[:, :, None]
        else:  # axes == 'xz':
            return msk if out_dim == 2 else msk[:, None, :]
    else:
        raise ValueError("invalid axes for differentiation")


def local_maxima_3d(
    arr: Float[Array, "s y x"],
) -> Int[Array, "n 3"]:
    """Find local maxima in a 3D scale space by checking whether a voxel is greater than all surrounding voxels.

    Args:
        arr (Float[Array, "s y x"]): array to find local maxima in.

    Returns:
        Float[Array, "n 3"]: coordinates of maxima
    """
    ss, sy, sx = arr.shape

    def _mask_wrapper(carry_idx: int, _) -> tuple[int, Bool[Array, "y x"]]:
        sub_arr = jax.lax.dynamic_slice(arr, [carry_idx, 0, 0], [3, sy, sx])
        return carry_idx + 1, _local_maxima_3d_mask(sub_arr)

    _, msk = jax.lax.scan(_mask_wrapper, 0, None, length=ss - 2)
    pks = jnp.stack(jnp.nonzero(msk), axis=1)
    return pks + 1


def _local_maxima_3d_mask(
    arr: Float[Array, "3 y x"],
) -> Bool[Array, "3 y x"]:
    sze_y, sze_x = arr.shape[1:]
    y, x = jnp.meshgrid(
        jnp.arange(1, sze_y - 1), jnp.arange(1, sze_x - 1), indexing="ij"
    )
    peak_mask = jax.vmap(Partial(_is_localmax_3d, arr), (0, 0), 0)(
        y.flatten(), x.flatten()
    ).reshape(y.shape)
    return peak_mask


def _is_localmax_3d(
    arr: Float[Array, "3 y x"],
    y: Int,
    x: Int,
) -> Bool:
    val = arr[1, y, x]
    nc = neighbor_coords_3d(1, y, x)
    neig_vals = arr[nc[:, 0], nc[:, 1], nc[:, 2]]
    return jnp.all(val > neig_vals)


def neighbor_coords_3d(z: Int, y: Int, x: Int) -> Int[Array, "26 3"]:
    """Get coordinates of all grid neighbors in 3D for the input grid coordinates.

    Volume being queried is assumed to be of shape (z, y, x)

    Args:
        z (int): z coordinate
        y (int): y coordinate
        x (int): x coordinate

    Returns:
        Int[Array, "26 3"]
    """
    return jnp.array(
        [
            [z - 1, y - 1, x - 1],
            [z - 1, y - 1, x],
            [z - 1, y - 1, x + 1],
            [z - 1, y, x - 1],
            [z - 1, y, x],
            [z - 1, y, x + 1],
            [z - 1, y + 1, x - 1],
            [z - 1, y + 1, x],
            [z - 1, y + 1, x + 1],
            [z, y - 1, x - 1],
            [z, y - 1, x],
            [z, y - 1, x + 1],
            [z, y, x - 1],
            [z, y, x + 1],
            [z, y + 1, x - 1],
            [z, y + 1, x],
            [z, y + 1, x + 1],
            [z + 1, y - 1, x - 1],
            [z + 1, y - 1, x],
            [z + 1, y - 1, x + 1],
            [z + 1, y, x - 1],
            [z + 1, y, x],
            [z + 1, y, x + 1],
            [z + 1, y + 1, x - 1],
            [z + 1, y + 1, x],
            [z + 1, y + 1, x + 1],
        ]
    )


def local_maxima_4d(
    arr: Float[Array, "s z y x"],
) -> Int[Array, "n 4"]:
    """Find local maxima in a 4D scale space by checking whether a voxel is greater than all surrounding voxels.

    Args:
        arr (Float[Array, "s z y x"]): array to find local maxima in.

    Returns:
        Float[Array, "n 4"]: coordinates of maxima
    """
    ss, sz, sy, sx = arr.shape

    def _mask_wrapper(carry_idx: int, _) -> tuple[int, Array]:
        sub_arr = jax.lax.dynamic_slice(
            arr, [carry_idx, 0, 0, 0], [3, sz, sy, sx]
        )
        return carry_idx + 1, _local_maxima_4d_mask(sub_arr)

    _, msk = jax.lax.scan(_mask_wrapper, 0, None, length=ss - 2)
    pks = jnp.stack(jnp.nonzero(msk), axis=1)
    return pks + 1


def _local_maxima_4d_mask(arr: Float[Array, "3 z y x"]) -> Bool[Array, "z y x"]:
    sze_z, sze_y, sze_x = arr.shape[1:]
    z, y, x = jnp.meshgrid(
        jnp.arange(1, sze_z - 1),
        jnp.arange(1, sze_y - 1),
        jnp.arange(1, sze_x - 1),
        indexing="ij",
    )
    peak_mask = jax.vmap(Partial(_is_localmax_4d, arr), (0, 0, 0), 0)(
        z.flatten(), y.flatten(), x.flatten()
    ).reshape(z.shape)
    return peak_mask


def _is_localmax_4d(
    arr: Float[Array, "3 z y x"], z: Int, y: Int, x: Int
) -> Bool:
    val = arr[1, z, y, x]
    nc = neighbor_coords_4d(1, z, y, x)
    return jnp.all(jax.vmap(lambda c: val > arr[c[0], c[1], c[2], c[3]])(nc))


def neighbor_coords_4d(s: Int, z: Int, y: Int, x: Int) -> Int[Array, "80 4"]:
    """Get coordinates of all grid neighbors in 4D for the input grid coordinates.

    Array being queried is assumed to be of shape (s, z, y, x)

    Args:
        s (int): s coordinate
        z (int): z coordinate
        y (int): y coordinate
        x (int): x coordinate

    Returns:
        Int[Array, "80 4"]
    """
    return jnp.array(
        [
            [s - 1, z - 1, y - 1, x - 1],
            [s - 1, z - 1, y - 1, x],
            [s - 1, z - 1, y - 1, x + 1],
            [s - 1, z - 1, y, x - 1],
            [s - 1, z - 1, y, x],
            [s - 1, z - 1, y, x + 1],
            [s - 1, z - 1, y + 1, x - 1],
            [s - 1, z - 1, y + 1, x],
            [s - 1, z - 1, y + 1, x + 1],
            [s - 1, z, y - 1, x - 1],
            [s - 1, z, y - 1, x],
            [s - 1, z, y - 1, x + 1],
            [s - 1, z, y, x - 1],
            [s - 1, z, y, x],
            [s - 1, z, y, x + 1],
            [s - 1, z, y + 1, x - 1],
            [s - 1, z, y + 1, x],
            [s - 1, z, y + 1, x + 1],
            [s - 1, z + 1, y - 1, x - 1],
            [s - 1, z + 1, y - 1, x],
            [s - 1, z + 1, y - 1, x + 1],
            [s - 1, z + 1, y, x - 1],
            [s - 1, z + 1, y, x],
            [s - 1, z + 1, y, x + 1],
            [s - 1, z + 1, y + 1, x - 1],
            [s - 1, z + 1, y + 1, x],
            [s - 1, z + 1, y + 1, x + 1],
            [s, z - 1, y - 1, x - 1],
            [s, z - 1, y - 1, x],
            [s, z - 1, y - 1, x + 1],
            [s, z - 1, y, x - 1],
            [s, z - 1, y, x],
            [s, z - 1, y, x + 1],
            [s, z - 1, y + 1, x - 1],
            [s, z - 1, y + 1, x],
            [s, z - 1, y + 1, x + 1],
            [s, z, y - 1, x - 1],
            [s, z, y - 1, x],
            [s, z, y - 1, x + 1],
            [s, z, y, x - 1],
            [s, z, y, x + 1],
            [s, z, y + 1, x - 1],
            [s, z, y + 1, x],
            [s, z, y + 1, x + 1],
            [s, z + 1, y - 1, x - 1],
            [s, z + 1, y - 1, x],
            [s, z + 1, y - 1, x + 1],
            [s, z + 1, y, x - 1],
            [s, z + 1, y, x],
            [s, z + 1, y, x + 1],
            [s, z + 1, y + 1, x - 1],
            [s, z + 1, y + 1, x],
            [s, z + 1, y + 1, x + 1],
            [s + 1, z - 1, y - 1, x - 1],
            [s + 1, z - 1, y - 1, x],
            [s + 1, z - 1, y - 1, x + 1],
            [s + 1, z - 1, y, x - 1],
            [s + 1, z - 1, y, x],
            [s + 1, z - 1, y, x + 1],
            [s + 1, z - 1, y + 1, x - 1],
            [s + 1, z - 1, y + 1, x],
            [s + 1, z - 1, y + 1, x + 1],
            [s + 1, z, y - 1, x - 1],
            [s + 1, z, y - 1, x],
            [s + 1, z, y - 1, x + 1],
            [s + 1, z, y, x - 1],
            [s + 1, z, y, x],
            [s + 1, z, y, x + 1],
            [s + 1, z, y + 1, x - 1],
            [s + 1, z, y + 1, x],
            [s + 1, z, y + 1, x + 1],
            [s + 1, z + 1, y - 1, x - 1],
            [s + 1, z + 1, y - 1, x],
            [s + 1, z + 1, y - 1, x + 1],
            [s + 1, z + 1, y, x - 1],
            [s + 1, z + 1, y, x],
            [s + 1, z + 1, y, x + 1],
            [s + 1, z + 1, y + 1, x - 1],
            [s + 1, z + 1, y + 1, x],
            [s + 1, z + 1, y + 1, x + 1],
        ]
    )
