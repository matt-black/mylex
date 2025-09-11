import math
from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..gauss import (
    point_source_image,
    point_source_volume,
    separable_gaussian_nd,
)
from ..util import fwhm_to_sigma

__all__ = [
    "ImageGaussian",
    "ModelGaussian",
    "VolumeGaussian",
]


class ModelGaussian(eqx.Module):

    _array: Float[Array, "y x"] | Float[Array, "z y x"]
    n_pts: int = eqx.field(static=True)
    centers: Float[Array, "n 3"] | Float[Array, "n 2"]
    amplitudes: Float[Array, " n"]
    sigma_lat: float = eqx.field(static=True)
    sigma_ax: float | None = eqx.field(static=True)

    def __init__(
        self,
        arr: Float[Array, "y x"] | Float[Array, "z y x"],
        centers: Float[Array, "n 3"],
        amplitudes: Float[Array, " n"],
        sigma_lat: float,
        sigma_ax: float | None = None,
    ) -> None:
        self._array = arr
        self.n_pts = self.centers.shape[0]
        self.centers = centers
        self.amplitudes = amplitudes
        self.sigma_lat = sigma_lat
        self.sigma_ax = sigma_ax

    @property
    @abstractmethod
    def point_source_channel(
        self,
    ) -> Float[Array, "y x"] | Float[Array, "z y x"]: ...

    @property
    def extended_source_channel(self):
        return self._array

    @property
    @abstractmethod
    def psf(self) -> Float[Array, "y x"] | Float[Array, "z y x"]: ...

    def __call__(self) -> Float[Array, "y x"] | Float[Array, "z y x"]:
        return jnp.add(self.point_source_channel, self.extended_source_channel)


class ImageGaussian(ModelGaussian):

    def __init__(
        self,
        img: Float[Array, "y x"],
        centers: Float[Array, "n 2"],
        amplitudes: Float[Array, " n"],
        fwhm_lat: float,
    ) -> None:
        return super().__init__(
            img, centers, amplitudes, fwhm_to_sigma(fwhm_lat), None
        )

    @property
    def point_source_channel(self):
        return point_source_image(
            self.sigma_lat, self.amplitudes, self.centers, *self._array.shape
        )

    @property
    def psf(self):
        width = math.ceil(self.sigma_lat * 4)
        width = width if width % 2 > 0 else width + 1
        array = separable_gaussian_nd(
            2,
            jnp.array([width // 2, width // 2], dtype=jnp.float32),
            jnp.array([self.sigma_lat, self.sigma_lat], dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
            *jnp.meshgrid(jnp.arange(width), jnp.arange(width)),
        )
        return array / jnp.sum(array)


class VolumeGaussian(ModelGaussian):

    def __init__(
        self,
        vol: Float[Array, "z y x"],
        centers: Float[Array, "n 2"],
        amplitudes: Float[Array, " n"],
        fwhm_lat: float,
        fwhm_ax: float,
    ) -> None:
        return super().__init__(
            vol,
            centers,
            amplitudes,
            fwhm_to_sigma(fwhm_lat),
            fwhm_to_sigma(fwhm_ax),
        )

    @property
    def point_source_channel(self):
        return point_source_volume(
            self.sigma_lat,
            self.sigma_ax,  # pyright: ignore[reportArgumentType]
            self.amplitudes,
            self.centers,
            *self.volume.shape,
        )

    @property
    def psf(self):
        width = math.ceil(self.sigma_lat * 4)
        width = width if width % 2 > 0 else width + 1
        height = math.ceil(
            self.sigma_ax * 4  # pyright: ignore[reportOptionalOperand]
        )
        height = height if height % 2 > 0 else height + 1
        array = separable_gaussian_nd(
            3,
            jnp.array([height // 2, width // 2, width // 2], dtype=jnp.float32),
            jnp.array(
                [self.sigma_ax, self.sigma_lat, self.sigma_lat],
                dtype=jnp.float32,
            ),
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
            *jnp.meshgrid(jnp.arange(width), jnp.arange(width)),
        )
        return array / jnp.sum(array)
