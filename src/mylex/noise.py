import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class AdditiveWhiteGaussianNoise(eqx.Module):
    """Module to add white gaussian noise to inputs."""

    sigma: float = eqx.field(static=True)

    def __init__(
        self,
        sigma: float,
    ):
        """Initialize the module.

        Args:
            sigma (float): std. deviation of Gaussian.
        """
        self.sigma = sigma

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Add Gaussian white noise to the input array.

        Args:
            x (Array): input array.
            key (PRNGKeyArray): PRNG key

        Returns:
            Array: input with added Gaussian white noise
        """
        noise = self.sigma * jax.random.normal(
            key, shape=x.shape, dtype=x.dtype
        )
        return x + noise


class SaltAndPepperNoise(eqx.Module):
    """Module to add salt & pepper impulse noise to inputs."""

    q_black: float = eqx.field(static=True)
    p_white: float = eqx.field(static=True)
    v_white: float = eqx.field(static=True)
    indep_channels: bool = eqx.field(static=True)
    drop_black: eqx.nn.Dropout
    drop_white: eqx.nn.Dropout

    def __init__(
        self,
        p_black: float,
        p_white: float,
        val_white: float,
        indep_channels: bool = False,
    ):
        """Initialize the salt & pepper noise generator.

        Args:
            p_black (float): probability of any pixel being set to black (0)
            p_white (float): probability of any pixel being set to white (max)
            val_white (float): value to set "white" pixels to.
            indep_channels (bool, optional): whether to apply noise independently to all channels or not. Defaults to False.
        """
        self.q_black = 1.0 - p_black
        self.p_white = p_white
        self.v_white = val_white
        self.drop_black = eqx.nn.Dropout(p_black)
        self.drop_white = eqx.nn.Dropout(1.0 - p_white)
        self.indep_channels = indep_channels

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Add salt & pepper noise to the input array.

        Args:
            x (Array): input array.
            key (PRNGKeyArray): PRNG key

        Returns:
            Array: noisy array
        """
        white_renorm = self.p_white * self.v_white
        if self.indep_channels:
            # do dropout with energy normalization
            m_black = (
                self.drop_black(jax.numpy.ones_like(x), key=key) * self.q_black
            )
            m_white = self.drop_white(x, key=key) * white_renorm
        else:
            m_black = jax.numpy.expand_dims(
                self.drop_black(jax.numpy.ones_like(x[0, ...]), key=key)
                * self.q_black,
                0,
            )
            m_white = jax.numpy.expand_dims(
                self.drop_white(x[0, ...], key=key) * white_renorm, 0
            )
        return jax.numpy.clip((x + m_white) * m_black, min=0, max=self.v_white)


@jax.custom_jvp
def shot_noise(key: PRNGKeyArray, arr: Array) -> Array:
    """Simulates Poisson shot noise.

    Args:
        key (PRNGKeyArray): PRNG key
        arr (Array): array to generate shot noise for

    Notes:
        The gradient is approximated using the gradient of a Gaussian.
        Taken from the chromatix library.
    """
    return jr.poisson(key, arr, arr.shape).astype(jnp.float32)


@shot_noise.defjvp
def shotnoise_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    key, arr = primals
    _, arr_dot = tangents
    primal_out = shot_noise(key, arr)
    # We define the gradient to be zero if image=0
    # we just add eta as we multiply by zero later anyway
    noise_grad = jnp.ones_like(arr) + jnp.divide(
        jr.normal(key, arr.shape), 2 * jnp.sqrt(arr) + 1e-6
    )
    # maximum operation, abs to get rid of -0)
    tangent_out = jnp.where(
        primal_out == 0, jnp.array(0.0), arr_dot * jnp.abs(noise_grad)
    )
    return primal_out, tangent_out
