import math

import equinox as eqx
import jax.numpy as jnp
import numpy
import optax
from jax.tree_util import Partial
from jaxtyping import Array, Float, PyTree
from tqdm.auto import trange

from ..resample import fourier_upsample
from ..util import fwhm_to_sigma, sigma_to_fwhm
from .loss import (
    ChiSquaredGaussian,
    FiredecMCSRegularization,
    MCSRegularization,
)
from .model import ImageGaussian, ModelGaussian, VolumeGaussian


def deconvolve(
    data: Float[Array, "y x"] | Float[Array, "z y x"],
    noise_map: Float[Array, "y x"] | Float[Array, "z y x"],
    centers: Float[Array, "n_pts 2"] | Float[Array, "n_pts 3"],
    amplitudes: Float[Array, " n_pts"],
    # pixel size & fwhm args
    pixel_size_data: float | tuple[float, float],
    upsample_ratio: int | tuple[int, int],
    fwhm_t: float | tuple[float, float],
    fwhm_r: float | tuple[float, float],
    # optimization arguments
    lagrange_multiplier: float,
    firedec_regularization: bool,
    optim: optax.GradientTransformationExtraArgs,
    n_iter: int,
    verbose: bool = False,
) -> tuple[ModelGaussian, numpy.ndarray]:

    if data.ndim == 2:  # image
        # make sure inputs are ok
        if noise_map.ndim != 2:
            raise ValueError("noise map must be same dimensions as data")
        if centers.shape[1] != 2:
            raise ValueError("centers must be (n_pts x 2)")
        if isinstance(fwhm_t, float):
            fwhm_t_ax, fwhm_t_lat = None, fwhm_t
        else:
            raise ValueError("fwhm_t must be single parameter, lateral")
        if isinstance(fwhm_r, float):
            fwhm_r_ax, fwhm_r_lat = None, fwhm_r
        else:
            raise ValueError("fwhm_r must be single parameter, lateral")
        # remap centers to model coordinates
        data_shape_y, data_shape_x = data.shape
        if isinstance(upsample_ratio, int):
            model_shape_y = data_shape_y * upsample_ratio
            model_shape_x = data_shape_x * upsample_ratio
            if isinstance(pixel_size_data, float):
                pixel_size_model = (
                    pixel_size_data / upsample_ratio,
                    pixel_size_data / upsample_ratio,
                )
            else:
                raise ValueError("pixel_size must be a single parameter")
            centers = centers @ jnp.diag(
                jnp.array([upsample_ratio, upsample_ratio])
            )
        else:
            raise ValueError("upsample_ratio must be single parameter")
        fwhm_s_apix = None
        model = ImageGaussian(
            fourier_upsample(data, (model_shape_y, model_shape_x)),
            centers,
            amplitudes,
            fwhm_r_lat,
        )
    elif data.ndim == 3:  # volume
        # make sure inputs are ok
        if noise_map.ndim != 3:
            raise ValueError("noise map must be same dimensions as data")
        if centers.shape[1] != 3:
            raise ValueError("centers must be (n_pts x 2)")
        if isinstance(fwhm_t, tuple):
            fwhm_t_ax, fwhm_t_lat = fwhm_t
        else:
            raise ValueError("must specify (ax, lat) fwhm")
        if isinstance(fwhm_r, tuple):
            fwhm_r_ax, fwhm_r_lat = fwhm_r
        else:
            raise ValueError("must specify (ax, lat) fwhm")
        # remap centers to model coordinates
        data_shape_z, data_shape_y, data_shape_x = data.shape
        if isinstance(upsample_ratio, tuple):
            model_shape_z = data_shape_z * upsample_ratio[0]
            model_shape_y = data_shape_y * upsample_ratio[1]
            model_shape_x = data_shape_x * upsample_ratio[1]
            if isinstance(pixel_size_data, tuple):
                pixel_size_model = (
                    pixel_size_data[0] / upsample_ratio[0],
                    pixel_size_data[1] / upsample_ratio[1],
                )
            else:
                raise ValueError("pixel_size must be a single parameter")
            centers = centers @ jnp.diag(
                jnp.array(
                    [upsample_ratio[0], upsample_ratio[1], upsample_ratio[1]]
                )
            )
        else:
            raise ValueError(
                "upsample_ratio must be specified (axial, lateral)"
            )
        # sigma_s is determined by the width of the t, r PSF's
        # since its convolved with the model data, specify its width in pixels relative to the upsampled model data
        sigma_t_ax = fwhm_to_sigma(fwhm_t_ax)
        sigma_r_ax = fwhm_to_sigma(fwhm_r_ax)
        sigma_s_ax = math.sqrt(sigma_t_ax**2 - sigma_r_ax**2)
        fwhm_s_apix = sigma_to_fwhm(sigma_s_ax) * (1 / pixel_size_model[0])
        # fwhm_r_apix = fwhm_r_ax * (1 / pixel_size_model[0])
        model = VolumeGaussian(
            fourier_upsample(
                data, (model_shape_z, model_shape_y, model_shape_x)
            ),
            centers,
            amplitudes,
            fwhm_r_lat,
            fwhm_r_ax,
        )
    else:
        raise ValueError("invalid input, must be image or volume")

    # convert full width, half max to std. dev
    # then determine the std. dev of each PSF in units of pixels
    # sigma_s is determined by the width of the t, r PSF's
    # since its convolved with the model data, specify its width in pixels relative to the upsampled model data
    sigma_t_lat = fwhm_to_sigma(fwhm_t_lat)
    sigma_r_lat = fwhm_to_sigma(fwhm_r_lat)
    sigma_s_lat = math.sqrt(sigma_t_lat**2 - sigma_r_lat**2)
    fwhm_s_lat = fwhm_to_sigma(sigma_s_lat)
    fwhm_s_lpix = fwhm_s_lat * (1 / pixel_size_model[1])
    fwhm_r_lpix = fwhm_r_lat * (1 / pixel_size_model[1])
    # formulate loss function
    chisqr = ChiSquaredGaussian(fwhm_s_lpix, fwhm_s_apix)
    regula = (
        FiredecMCSRegularization(fwhm_r_lpix, fwhm_s_apix)
        if firedec_regularization
        else MCSRegularization(fwhm_s_lpix, fwhm_s_apix)
    )

    def _loss(
        d: Float[Array, "y x"],  # data
        nm: Float[Array, "y x"],  # noise map
        m: ModelGaussian,  # model
        chi: ChiSquaredGaussian,  # chi-squared loss
        reg: (
            MCSRegularization | FiredecMCSRegularization
        ),  # regularization term
        lag_mult: float,  # lagrange multiplier
    ) -> Float[Array, ""]:
        return chi(d, nm, m) + lag_mult * reg(m)

    loss = eqx.filter_jit(
        Partial(
            _loss,
            data,
            noise_map,
            chi=chisqr,
            reg=regula,
            lag_mult=lagrange_multiplier,
        )
    )

    def _make_step(
        m: ModelGaussian, opt_state: PyTree
    ) -> tuple[ModelGaussian, optax.OptState, Float[Array, ""]]:
        loss_val, grads = eqx.filter_value_and_grad(loss)(m)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(m, eqx.is_array)
        )
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss_val

    losses = numpy.zeros((n_iter,), dtype=numpy.float32)
    prog_bar = (
        trange(n_iter, desc="Optimizing", leave=False)
        if verbose
        else range(n_iter)
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for step_idx in prog_bar:
        model, opt_state, loss_val = _make_step(model, opt_state)
        losses[step_idx] = loss_val
        if verbose:
            prog_bar.set_postfix(  # pyright: ignore[reportAttributeAccessIssue]
                {"loss": loss_val}
            )
    return model, losses
