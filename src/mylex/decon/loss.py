import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jaxtyping import Array, Float

from ..gauss import separable_gaussian_nd
from ..resample import fourier_downsample
from ..util import fwhm_to_sigma
from .model import ModelGaussian


class _PSFLoss(eqx.Module):

    fwhm_lat: float = eqx.field(static=True)
    fwhm_ax: float | None = eqx.field(static=True)
    psf: Float[Array, "y x"] | Float[Array, "z y x"]

    def __init__(
        self,
        fwhm_lat: float,
        fwhm_ax: float | None,
    ) -> None:
        self.fwhm_lat = fwhm_lat
        self.fwhm_ax = fwhm_ax
        # compute the psf
        sigma_lat = fwhm_to_sigma(fwhm_lat)
        width = math.ceil(sigma_lat * 3)
        width = width + 1 if width % 2 == 0 else width
        if fwhm_ax is None:
            # must be 2D
            self.psf = separable_gaussian_nd(
                jnp.array([width // 2, width // 2], dtype=jnp.float32),
                jnp.array([sigma_lat, sigma_lat], dtype=jnp.float32),
                jnp.array([1.0], dtype=jnp.float32),
                jnp.array([0.0], dtype=jnp.float32),
                *jnp.meshgrid(jnp.arange(width), jnp.arange(width)),
            )
        else:  # 3D
            sigma_ax = fwhm_to_sigma(fwhm_ax)
            height = math.ceil(sigma_ax * 3)
            height = height if height % 2 > 0 else height + 1
            self.psf = separable_gaussian_nd(
                jnp.array(
                    [height // 2, width // 2, width // 2], dtype=jnp.float32
                ),
                jnp.array([sigma_ax, sigma_lat, sigma_lat], dtype=jnp.float32),
                jnp.array([1.0], dtype=jnp.float32),
                jnp.array([0.0], dtype=jnp.float32),
                *jnp.meshgrid(jnp.arange(width), jnp.arange(width)),
            )


class ChiSquaredGaussian(_PSFLoss):

    def __init__(
        self,
        fwhm_lat: float,
        fwhm_ax: float | None,
    ) -> None:
        return super().__init__(fwhm_lat, fwhm_ax)

    def __call__(
        self,
        data: Float[Array, "y x"] | Float[Array, "z y x"],
        noise_map: Float[Array, "y x"] | Float[Array, "z y x"],
        model: ModelGaussian,
    ) -> Float[Array, ""]:
        # convolve the model with the PSF, then downsample it so it matches the size of the data
        pred = fourier_downsample(
            convolve(model(), jax.lax.stop_gradient(self.psf), mode="same"),
            data.shape,
        )
        N = math.prod(pred.shape)
        # residuals are squared differences normalized by the noise map
        return jnp.sum(jnp.divide(jnp.square(data - pred), noise_map)) / N


class MCSRegularization(_PSFLoss):

    def __init__(
        self,
        fwhm_lat: float,
        fwhm_ax: float | None,
    ) -> None:
        return super().__init__(fwhm_lat, fwhm_ax)

    def __call__(
        self,
        model: ModelGaussian,
    ) -> Float[Array, ""]:
        blr = convolve(
            model.extended_source_channel,
            jax.lax.stop_gradient(self.psf),
            mode="same",
        )
        N = math.prod(blr.shape)
        return (
            jnp.sum(
                jnp.square(jnp.subtract(model.extended_source_channel, blr))
            )
            / N
        )


class FiredecMCSRegularization(_PSFLoss):

    def __init__(
        self,
        fwhm_lat: float,
        fwhm_ax: float | None,
    ) -> None:
        return super().__init__(fwhm_lat, fwhm_ax)

    def __call__(self, model: ModelGaussian) -> Float[Array, ""]:
        blr = convolve(
            model.extended_source_channel,
            jax.lax.stop_gradient(self.psf),
            mode="same",
        )
        N = math.prod(blr.shape)
        return (
            jnp.sum(
                jnp.divide(
                    jnp.subtract(model.extended_source_channel, blr),
                    jnp.sqrt(jnp.add(1, model.extended_source_channel)),
                )
            )
            / N
        )
