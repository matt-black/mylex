import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray


class BernoulliMaskMaker(eqx.Module):
    """Module for generating masks for inputs to denoising NNs by Bernoulli sampling pixels/voxel coordinates."""

    dropout: eqx.nn.Dropout
    q_mask: float = eqx.field(static=True)
    indep_channels: bool = eqx.field(static=True)

    def __init__(
        self,
        p_mask: float,
        indep_channels: bool,
    ):
        """Initialize the masking module.

        Args:
            p_mask (float): probability a pixel is masked out.
            indep_channels (bool): whether the channels should be independently masked. if False, a single-channel mask will be generated and applied to all channels, if True then all pixels/voxels in the input are treated independently when masking. Defaults to False.
        """
        self.q_mask = 1.0 - p_mask
        self.indep_channels = indep_channels
        self.dropout = eqx.nn.Dropout(p_mask)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Generate a Bernoulli-sampled mask for the input.

        Args:
            x (Array): input array, to be masked
            key (PRNGKeyArray): PRNG key

        Returns:
            Array: Mask for the input array. The mask is one where the data should be kept, and zero where it should be zero'd.
        """
        if self.indep_channels:
            return self.dropout(
                jax.numpy.ones_like(x), key=key
            )  # * self.q_mask
        else:
            # generate a mask of a single channel, then repeat it so that all channels are masked in the same way
            mask = self.dropout(
                jax.numpy.ones_like(x, shape=[1] + list(x.shape[1:])), key=key
            )
            return mask.repeat(x.shape[0], axis=0)
