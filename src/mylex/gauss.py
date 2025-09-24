import math

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.tree_util import Partial
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from scipy.special import erfcinv

from .noise import shot_noise
from .util import l2_loss


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


def source_image(
    sigmas: Float[Array, " n_pts"],
    amplitudes: Float[Array, " n_pts"],
    centers: Float[Array, "n_pts 2"],
    background: Float[Array, ""],
    shape_y: int,
    shape_x: int,
    checkpoint: bool = False,
    extent: int | float | None = None,
) -> Float[Array, "{shape_y} {shape_x}"]:
    grid = jnp.meshgrid(jnp.arange(shape_y), jnp.arange(shape_x), indexing="ij")

    if extent is None:
        gauss_fun = gauss_2d
    else:
        gauss_fun = Partial(gauss_2d_limited, extent=extent)

    def add_source(
        curr: Float[Array, "{shape_y} {shape_x}"],
        params: Float[Array, " 4"],
    ) -> tuple[Float[Array, "{shape_y} {shape_x}"], None]:
        amp, sig, cy, cx = params[0], params[1], params[2], params[3]
        return (
            jnp.add(
                curr,
                jnp.multiply(
                    amp, gauss_fun(cy, cx, sig, sig, grid[0], grid[1])
                ),
            ),
            None,
        )

    if checkpoint:
        scan_fun = jax.checkpoint(  # pyright: ignore[reportPrivateImportUsage]
            add_source
        )
    else:
        scan_fun = add_source

    params = jnp.concatenate(
        [amplitudes[:, None], sigmas[:, None], centers], axis=1
    )

    out, _ = jax.lax.scan(
        scan_fun,
        jnp.zeros_like(grid[0], dtype=jnp.float32),
        params,
    )
    return out + background


def point_source_image(
    sigma: float,
    amplitudes: Float[Array, " n_pts"],
    centers: Float[Array, "n_pts 2"],
    shape_y: int,
    shape_x: int,
    checkpoint: bool = False,
    extent: int | float | None = None,
) -> Float[Array, "{shape_y} {shape_x}"]:
    return source_image(
        jnp.ones_like(amplitudes) * sigma,
        amplitudes,
        centers,
        jnp.array(0.0),
        shape_y,
        shape_x,
        checkpoint,
        extent,
    )


def source_volume(
    sigma_lat: Float[Array, " n_pts"],
    sigma_ax: Float[Array, " n_pts"],
    amplitudes: Float[Array, " n_pts"],
    centers: Float[Array, "n_pts 3"],
    background: Float[Array, ""],
    shape_z: int,
    shape_y: int,
    shape_x: int,
    checkpoint: bool = False,
    extent: int | float | None = None,
) -> Float[Array, "{shape_z} {shape_y} {shape_x}"]:
    grid = jnp.meshgrid(
        jnp.arange(shape_z),
        jnp.arange(shape_y),
        jnp.arange(shape_x),
        indexing="ij",
    )
    if extent is None:
        gauss_fun = gauss_3d
    else:
        gauss_fun = Partial(gauss_3d_limited, extent=extent)

    def add_source(
        curr: Float[Array, "{shape_y} {shape_x}"],
        params: Float[Array, " 6"],
    ) -> tuple[Float[Array, "{shape_y} {shape_x}"], None]:
        syx, sz = params[0], params[1]
        amp, cz, cy, cx = params[2], params[3], params[4], params[5]
        g = gauss_fun(cz, cy, cx, sz, syx, syx, grid[0], grid[1], grid[2])
        return jnp.add(curr, jnp.multiply(amp, g)), None

    if checkpoint:
        scan_fun = jax.checkpoint(  # pyright: ignore[reportPrivateImportUsage]
            add_source
        )
    else:
        scan_fun = add_source

    params = jnp.concatenate(
        [amplitudes[:, None], sigma_lat[:, None], sigma_ax[:, None], centers],
        axis=1,
    )
    out, _ = jax.lax.scan(
        scan_fun,
        jnp.zeros_like(grid[0], dtype=jnp.float32),
        params,
    )
    return out + background


def point_source_volume(
    sigma_lat: float,
    sigma_ax: float,
    amplitudes: Float[Array, " n_pts"],
    centers: Float[Array, "n_pts 3"],
    shape_z: int,
    shape_y: int,
    shape_x: int,
    checkpoint: bool = False,
    extent: int | float | None = None,
) -> Float[Array, "{shape_z} {shape_y} {shape_x}"]:
    return source_volume(
        jnp.ones_like(amplitudes) * sigma_lat,
        jnp.ones_like(amplitudes) * sigma_ax,
        amplitudes,
        centers,
        jnp.array(0.0),
        shape_z,
        shape_y,
        shape_x,
        checkpoint,
        extent,
    )


def fit_source_volume(
    centers: Float[Array, "n_pts 3"],
    amplitudes: Float[Array, " n_pts"],
    sigma_lats: Float[Array, " n_pts"],
    sigma_axs: Float[Array, " n_pts"],
    background: Float[Array, ""],
    target: Float[Array, "z y x"],
    num_steps: int,
    key: PRNGKeyArray,
    fix_sigma: bool = False,
    extent: int | float | None = None,
    learning_rate: float = 1e-1,
) -> tuple[
    Float[Array, "n_pts 6"], Float[Array, ""], Float[Array, " {num_steps}"]
]:
    n_pts = amplitudes.shape[0]
    shape_z, shape_y, shape_x = target.shape

    # setup source parameters and optimization function
    if fix_sigma:
        src_par_i = jnp.concatenate([amplitudes[:, None], centers], axis=1)

        def opt_fun(params: Float[Array, " n_pts*4+1"], key: PRNGKeyArray):
            src_par = jnp.reshape(params[:-1], (n_pts, 4))
            bkgrnd = params[-1]
            pred = source_volume(
                sigma_lats,
                sigma_axs,
                src_par[:, 0],
                src_par[:, 1:],
                bkgrnd,
                shape_z,
                shape_y,
                shape_x,
                checkpoint=True,
                extent=extent,
            )
            noisy_pred = shot_noise(key, pred)
            return jnp.mean(l2_loss(noisy_pred, target))

    else:
        src_par_i = jnp.concatenate(
            [
                sigma_lats[:, None],
                sigma_axs[:, None],
                amplitudes[:, None],
                centers,
            ],
            axis=1,
        )

        def opt_fun(params: Float[Array, " n_pts*6+1"], key: PRNGKeyArray):
            src_par = jnp.reshape(params[:-1], (n_pts, 6))
            bkgrnd = params[-1]
            pred = source_volume(
                src_par[:, 0],
                src_par[:, 1],
                src_par[:, 2],
                src_par[:, 3:],
                bkgrnd,
                shape_z,
                shape_y,
                shape_x,
                checkpoint=True,
                extent=extent,
            )
            noisy_pred = shot_noise(key, pred)
            return jnp.mean(l2_loss(noisy_pred, target))

    # initialize optimization
    par_i = jnp.concatenate([src_par_i.flatten(), background[None]], axis=0)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(par_i)

    def make_step(
        carry: tuple[Array, PyTree],
        key: PRNGKeyArray,
    ) -> tuple[tuple[Array, PyTree], Float[Array, ""]]:
        params, opt_state = carry
        loss, grads = jax.value_and_grad(opt_fun)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params: Array = optax.apply_updates(
            params, updates
        )  # pyright: ignore[reportAssignmentType]
        return (params, opt_state), loss

    (par_f, opt_state), losses = jax.lax.scan(
        make_step,
        (par_i, opt_state),
        jr.split(key, num_steps),
        length=num_steps,
    )
    src_par = par_f[:-1].reshape(n_pts, (4 if fix_sigma else 6))
    if fix_sigma:
        src_par = jnp.concatenate(
            [sigma_lats[:, None], sigma_axs[:, None], src_par], axis=1
        )
    background = par_f[-1]
    return src_par, background, losses


def separable_gaussian_nd(
    cent: Float[Array, " {n_dim}"],
    sigma: Float[Array, " {n_dim}"],
    amplitude: Float[Array, ""],
    background: Float[Array, ""],
    *coords,
) -> Array:
    c = jnp.stack(coords, axis=0)
    g = jnp.prod(jax.vmap(gauss_1d, (0, 0, 0), 0)(cent, sigma, c), axis=0)
    return jnp.add(jnp.multiply(amplitude, g), background)


@jax.custom_jvp
def gauss_1d(
    mu: Float[Array, ""],
    sigma: Float[Array, ""],
    x: Array,
) -> Array:
    return jnp.multiply(
        jnp.divide(1.0, jnp.sqrt(2 * jnp.pi) * sigma),
        jnp.exp(
            jnp.negative(jnp.divide(jnp.square(x - mu), 2 * jnp.square(sigma)))
        ),
    )


@gauss_1d.defjvp
def gauss_1d_jvp(
    primals: tuple[Array, Array, Array], tangents: tuple[Array, Array, Array]
) -> tuple[Array, Array]:
    mu, sigma, x = primals
    mu_dot, sigma_dot, _ = tangents
    primal_out = gauss_1d(mu, sigma, x)
    var = jnp.square(sigma)
    dmu = jnp.divide(x - mu, var)
    dsig = jnp.divide(jnp.square(x - mu) - var, jnp.power(sigma, 3))
    return primal_out, (dmu * mu_dot + dsig * sigma_dot)


@jax.custom_jvp
def gauss_1d_limited(
    mu: Float[Array, ""],
    sigma: Float[Array, ""],
    x: Array,
    extent: int | float = 6,
) -> Array:
    var = jnp.square(sigma)
    return jnp.where(
        jnp.square(x - mu) < extent * sigma,
        (1 / (jnp.sqrt(2 * jnp.pi) * sigma))
        * jnp.exp(jnp.negative(jnp.divide(jnp.square(x - mu), 2 * var))),
        0.0,
    )


@gauss_1d_limited.defjvp
def gauss_1d_limited_jvp(
    primals: tuple[Array, Array, Array, int | float],
    tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    mu, sigma, x, extent = primals
    mu_dot, sigma_dot, _, _ = tangents
    primal_out = gauss_1d_limited(mu, sigma, x, extent)
    dmu = (x - mu) / jnp.square(sigma)
    dsig = jnp.divide(
        jnp.square(x - mu) - jnp.square(sigma), jnp.power(sigma, 3)
    )
    return primal_out, jnp.where(
        jnp.square(x - mu) < extent * sigma,
        (dmu * mu_dot + dsig * sigma_dot) * primal_out,
        0.0,
    )


@jax.custom_jvp
def gauss_2d(
    mu_y: Float[Array, ""],
    mu_x: Float[Array, ""],
    sd_y: Float[Array, ""],
    sd_x: Float[Array, ""],
    y: Array,
    x: Array,
) -> Array:
    return jnp.multiply(gauss_1d(mu_y, sd_y, y), gauss_1d(mu_x, sd_x, x))


@gauss_2d.defjvp
def gauss_2d_jvp(
    primals: tuple[Array, Array, Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array, Array, Array],
) -> tuple[Array, Array]:
    my, mx, sy, sx, y, x = primals
    myd, mxd, syd, sxd, _, _ = tangents
    p_y = gauss_1d(my, sy, y)
    p_x = gauss_1d(mx, sx, x)
    primal_out = jnp.multiply(p_y, p_x)
    # derivatives
    dmx = (x - mx) / jnp.square(sx)
    dsx = jnp.divide(jnp.square(x - mx) - jnp.square(sx), jnp.power(sx, 3))
    dmy = (y - my) / jnp.square(sy)
    dsy = jnp.divide(jnp.square(y - my) - jnp.square(sy), jnp.power(sy, 3))
    tangent_out = primal_out * (dmy * myd + dsy * syd + dmx * mxd + dsx * sxd)
    return primal_out, tangent_out


@jax.custom_jvp
def gauss_2d_limited(
    mu_y: Float[Array, ""],
    mu_x: Float[Array, ""],
    sd_y: Float[Array, ""],
    sd_x: Float[Array, ""],
    y: Array,
    x: Array,
    extent: int | float = 6,
) -> Array:
    return jnp.where(
        jnp.logical_and(
            jnp.square(x - mu_x) < sd_x * extent,
            jnp.square(y - mu_y) < sd_y * extent,
        ),
        gauss_2d(mu_y, sd_y, y, mu_x, sd_x, x),
        0.0,
    )


@gauss_2d_limited.defjvp
def gauss_2d_limited_jvp(
    primals: tuple[Array, Array, Array, Array, Array, Array, int | float],
    tangents: tuple[Array, Array, Array, Array, Array, Array, Array],
) -> tuple[Array, Array]:
    my, mx, sy, sx, y, x, ext = primals
    myd, mxd, syd, sxd, _, _, _ = tangents
    gauss_ext = Partial(gauss_1d_limited, extent=ext)
    p_y = gauss_ext(my, sy, y)
    p_x = gauss_ext(mx, sx, x)
    primal_out = jnp.multiply(p_y, p_x)
    # derivatives
    dmx = (x - mx) / jnp.square(sx)
    dsx = jnp.divide(jnp.square(x - mx) - jnp.square(sx), jnp.power(sx, 3))
    dmy = (y - my) / jnp.square(sy)
    dsy = jnp.divide(jnp.square(y - my) - jnp.square(sy), jnp.power(sy, 3))
    return primal_out, jnp.where(
        jnp.logical_and(
            jnp.square(y - my) < ext * sy, jnp.square(x - mx) < ext * sx
        ),
        (dmy * myd + dsy * syd + dmx * mxd + dsx * sxd) * primal_out,
        0.0,
    )


@jax.custom_jvp
def gauss_3d(
    mu_z: Float[Array, ""],
    mu_y: Float[Array, ""],
    mu_x: Float[Array, ""],
    sd_z: Float[Array, ""],
    sd_y: Float[Array, ""],
    sd_x: Float[Array, ""],
    z: Array,
    y: Array,
    x: Array,
) -> Array:
    return jnp.multiply(
        jnp.multiply(gauss_1d(mu_y, sd_y, y), gauss_1d(mu_x, sd_x, x)),
        gauss_1d(mu_z, sd_z, z),
    )


@gauss_3d.defjvp
def gauss_3d_jvp(
    primals: tuple[
        Array, Array, Array, Array, Array, Array, Array, Array, Array
    ],
    tangents: tuple[
        Array, Array, Array, Array, Array, Array, Array, Array, Array
    ],
) -> tuple[Array, Array]:
    mz, my, mx, sz, sy, sx, z, y, x = primals
    mzd, myd, mxd, szd, syd, sxd, _, _, _ = tangents
    p_z = gauss_1d(mz, sz, z)
    p_y = gauss_1d(my, sy, y)
    p_x = gauss_1d(mx, sx, x)
    primal_out = p_z * p_y * p_x
    # derivatives
    dmx = (x - mx) / jnp.square(sx)
    dsx = jnp.divide(jnp.square(x - mx) - jnp.square(sx), jnp.power(sx, 3))
    dmy = (y - my) / jnp.square(sy)
    dsy = jnp.divide(jnp.square(y - my) - jnp.square(sy), jnp.power(sy, 3))
    dmz = (z - mz) / jnp.square(sz)
    dsz = jnp.divide(jnp.square(z - mz) - jnp.square(sz), jnp.power(sz, 3))
    tangent_out = primal_out * (
        dmz * mzd + dsz * szd + dmy * myd + dsy * syd + dmx * mxd + dsx * sxd
    )
    return primal_out, tangent_out


@jax.custom_jvp
def gauss_3d_limited(
    mu_z: Float[Array, ""],
    mu_y: Float[Array, ""],
    mu_x: Float[Array, ""],
    sd_z: Float[Array, ""],
    sd_y: Float[Array, ""],
    sd_x: Float[Array, ""],
    z: Array,
    y: Array,
    x: Array,
    extent: int | float = 6,
) -> Array:
    return jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                jnp.square(x - mu_x) < sd_x * extent,
                jnp.square(y - mu_y) < sd_y * extent,
            ),
            jnp.square(z - mu_z) < sd_z * extent,
        ),
        gauss_3d(mu_z, mu_y, mu_x, sd_z, sd_y, sd_x, z, y, x),
        0.0,
    )


@gauss_3d_limited.defjvp
def gauss_3d_limited_jvp(
    primals: tuple[Array, ...], tangents: tuple[Array, ...]
) -> tuple[Array, Array]:
    mz, my, mx, sz, sy, sx, z, y, x, ext = primals
    mzd, myd, mxd, szd, syd, sxd, _, _, _, _ = tangents
    gauss_ext = Partial(gauss_1d_limited, extent=ext)
    p_z = gauss_ext(mz, sz, z)
    p_y = gauss_ext(my, sy, y)
    p_x = gauss_ext(mx, sx, x)
    primal_out = p_z * p_y * p_x
    # derivatives
    dmx = (x - mx) / jnp.square(sx)
    dsx = jnp.divide(jnp.square(x - mx) - jnp.square(sx), jnp.power(sx, 3))
    dmy = (y - my) / jnp.square(sy)
    dsy = jnp.divide(jnp.square(y - my) - jnp.square(sy), jnp.power(sy, 3))
    dmz = (z - mz) / jnp.square(sz)
    dsz = jnp.divide(jnp.square(z - mz) - jnp.square(sz), jnp.power(sz, 3))
    return primal_out, jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                jnp.square(z - mz) < ext * sz, jnp.square(y - my) < ext * sy
            ),
            jnp.square(x - mx) < ext * sx,
        ),
        (dmz * mzd + dsz * szd + dmy * myd + dsy * syd + dmx * mxd + dsx * sxd)
        * primal_out,
        0.0,
    )  # pyright: ignore[reportReturnType]
