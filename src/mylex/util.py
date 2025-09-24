import math
from itertools import product

import jax.numpy as jnp
from jaxtyping import Array, Float
from optax.losses import l2_loss as optax_l2


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


def lateral_resolution_confocal(na: float, _lambda: float = 550) -> float:
    """Compute the lateral resolution of a confocal microscope with the specified numerical aperture objective and at the specified wavelength.

    Args:
        na (float): numerical aperture of the objective
        _lambda (float): wavelength of emission light

    Returns:
        float
    """
    return 0.61 * _lambda / na


def axial_resolution_confocal(
    na: float, _lambda: float = 550, n: float = 1.34
) -> float:
    """Compute the axial resolution of a confocal microscope with the specified numerical aperture objective and at the specified wavelength.

    Args:
        na (float): numerical aperture of the objective
        _lambda (float): wavelength of emission light

    Returns:
        float
    """
    return 2 * _lambda * n / na**2


def poisson_nll(pred: Array, target: Array) -> Array:
    eps = jnp.finfo(pred.dtype).eps
    stir = (
        target * jnp.log(target + eps)
        - target
        + 0.5 * jnp.log(2 * jnp.pi * target)
    )
    return pred - target * jnp.log(pred + eps) + stir


def l2_loss(pred: Array, target: Array) -> Array:
    return jnp.mean(optax_l2(pred, target))


def output_shape_for_transform(
    t: Float[Array, "4 4"],
    input_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Calculate output shape of transformed volume.

    Args:
        t (NDArray): affine transform matrix
        input_shape (Iterable): shape of input volume (ZRC)

    Returns:
        Tuple[int,int,int]: output shape (ZRC)
    """
    coord = list(product(*[(0, s) for s in input_shape[::-1]]))
    coord = jnp.asarray(coord).T
    coord = jnp.vstack([coord, jnp.zeros_like(coord[0, :])])
    coordT = (t @ coord)[:-1, :]
    ptp = jnp.ceil(jnp.ptp(coordT, axis=1))
    return tuple(
        [int(v) for v in ptp[::-1]]
    )  # pyright: ignore[reportReturnType]


def output_shape_for_inv_transform(
    T: Array, input_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Calculate output shape of (inverse)-transformed volume.

    Args:
        T (NDArray): affine transform matrix (to be inverted)
        input_shape (Iterable): shape of input volume (ZRC)

    Returns:
        Tuple[int,int,int]: shape of output volume (ZRC)
    """

    fwd = jnp.linalg.inv(T)
    return output_shape_for_transform(fwd, input_shape)


def _yaw_matrix(alpha):
    return jnp.vstack(
        [
            jnp.array([math.cos(alpha), -math.sin(alpha), 0]),
            jnp.array([math.sin(alpha), math.cos(alpha), 0]),
            jnp.array([0, 0, 1]),
        ]
    )


def _pitch_matrix(beta):
    return jnp.vstack(
        [
            jnp.array([math.cos(beta), 0, math.sin(beta)]),
            jnp.array([0, 1, 0]),
            jnp.array([-math.sin(beta), 0, math.cos(beta)]),
        ]
    )


def _roll_matrix(gamma):
    return jnp.vstack(
        [
            jnp.array([1, 0, 0]),
            jnp.array([0, math.cos(gamma), -math.sin(gamma)]),
            jnp.array([0, math.sin(gamma), math.cos(gamma)]),
        ]
    )


def rotation_matrix(
    alpha: float, beta: float, gamma: float
) -> Float[Array, "4 4"]:
    yaw = _yaw_matrix(alpha)
    pitch = _pitch_matrix(beta)
    roll = _roll_matrix(gamma)
    R = yaw @ pitch @ roll
    return jnp.concatenate(
        [
            jnp.concatenate([R, jnp.asarray([0, 0, 0])[:, None]], axis=1),
            jnp.asarray([0, 0, 0, 1])[None, :],
        ],
        axis=0,
    )


def rotation_about_point_matrix(
    alpha: float, beta: float, gamma: float, x: float, y: float, z: float
) -> Float[Array, "4 4"]:
    t1 = translation_matrix(x, y, z)
    R = rotation_matrix(alpha, beta, gamma)
    t2 = translation_matrix(-x, -y, -z)
    return t1 @ R @ t2


def translation_matrix(x: float, y: float, z: float) -> Float[Array, "4 4"]:
    return jnp.asarray([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def scale_matrix(sx: float, sy: float, sz: float) -> Float[Array, "4 4"]:
    return jnp.diag(jnp.asarray([sx, sy, sz, 1.0]))
