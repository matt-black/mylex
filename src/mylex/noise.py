import equinox as eqx
import jax
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
