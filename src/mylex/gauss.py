import math

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
