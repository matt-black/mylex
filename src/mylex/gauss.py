import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.special import erfcinv


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


def truncation_error_nd_to_1d(epsilon: float, dim: int) -> float:
    """Convert relative truncation error over `dim`-dimensional domain to a relative truncation error over a 1D domain.

    Args:
        epsilon (float): N-dimensional relative truncation error
        dim (int): number of dimensions (N)

    Returns:
        float
    """
    return 1 - (1 - epsilon) ** (1 / dim)


def point_source_image(
    sigma: float,
    amplitudes: Float[Array, " n_pts"],
    centers: Float[Array, "n_pts 2"],
    shape_y: int,
    shape_x: int,
) -> Float[Array, "{shape_y} {shape_x}"]:
    grid = jnp.meshgrid(
        jax.lax.stop_gradient(jnp.arange(shape_x)),
        jax.lax.stop_gradient(jnp.arange(shape_y)),
        indexing="xy",
    )

    def _wrap_gauss(
        center: Float[Array, " 2"], amplitude: Float[Array, ""]
    ) -> Float[Array, "{shape_y} {shape_x}"]:
        return separable_gaussian_nd(
            2,
            center,
            jax.lax.stop_gradient(jnp.array([sigma, sigma])),
            amplitude,
            jnp.array([0.0]),
            *grid,
        )

    return jnp.sum(
        jax.vmap(_wrap_gauss, (0, 0), 0)(centers, amplitudes), axis=0
    )


def point_source_volume(
    sigma_lat: float,
    sigma_ax: float,
    amplitudes: Float[Array, " n_pts"],
    centers: Float[Array, "n_pts 3"],
    shape_z: int,
    shape_y: int,
    shape_x: int,
) -> Float[Array, "{shape_z} {shape_y} {shape_x}"]:
    grid = jnp.meshgrid(
        jax.lax.stop_gradient(jnp.arange(shape_z)),
        jax.lax.stop_gradient(jnp.arange(shape_x)),
        jax.lax.stop_gradient(jnp.arange(shape_y)),
        indexing="xy",
    )

    def _wrap_gauss(
        center: Float[Array, " 3"], amplitude: Float[Array, ""]
    ) -> Float[Array, "{shape_z} {shape_y} {shape_x}"]:
        return separable_gaussian_nd(
            3,
            center,
            jax.lax.stop_gradient(jnp.array([sigma_ax, sigma_lat, sigma_lat])),
            amplitude,
            jnp.array([0.0]),
            *grid,
        )

    return jnp.sum(
        jax.vmap(_wrap_gauss, (0, 0), 0)(centers, amplitudes), axis=0
    )


def separable_gaussian_nd(
    n_dim: int,
    cent: Float[Array, " {n_dim}"],
    sigma: Float[Array, " {n_dim}"],
    amplitude: Float[Array, ""],
    background: Float[Array, ""],
    *coords,
) -> Array:
    # setup the broadcasting so that center & sigma will be applied to each coord
    exp_dims = list(range(1, n_dim + 1))
    cent = jax.lax.expand_dims(cent, exp_dims)
    sigma = jax.lax.expand_dims(sigma, exp_dims)
    # formulate the term inside the exponential -- this is effectively each 1D gaussian on each axis
    exp_term = jnp.sum(
        __gauss_term(cent, sigma, jnp.stack(coords, axis=0)), axis=0
    )
    # multiply by amplitude, add background
    return amplitude * jnp.exp(jnp.negative(exp_term)) + background


def __gauss_term(c: Float[Array, ""], sx: Float[Array, ""], x: Array) -> Array:
    return jnp.square(x - c) / (2 * jnp.square(sx))
