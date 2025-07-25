from collections.abc import Callable
from typing import Sequence, Tuple, Union

import equinox as eqx
import jax
from jax.tree_util import Partial
from jaxtyping import Array, PRNGKeyArray

from .conv import ConvBlock


def upsample_2x(x: Array, num_spatial_dims: int, method: str) -> Array:
    """Upsample the input array to twice its original size.

    Args:
        x (Array): input array, to be upsampled
        num_spatial_dims (int): # of spatial dimensions in the input (assumed trailing)
        method (str): interpolation method for upsampling.

    Returns:
        Array
    """
    output_shape = [s for s in x.shape[:-num_spatial_dims]] + [
        2 * s for s in x.shape[-num_spatial_dims:]
    ]
    return jax.image.resize(x, shape=output_shape, method=method)


class UpBlock(eqx.Module):
    """UpBlock Decoder block for use in a UNet.

    Takes in the current array being passed through the network as well as an "encoding" that comes from a skip connection from the encoder.
    """

    block: ConvBlock
    upsample: Union[eqx.nn.ConvTranspose, Callable[[Array], Array]]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: Union[int, Tuple[int, int]],
        upsampling_mode: str,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        activation: str = "leaky_relu",
        dropout_prob: float = 0.3,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the decoder block.

        Args:
            num_spatial_dims (int): # of spatial dims in input arrays (assumed trailing)
            in_channels (int): # of input channels
            out_channels (Union[int, Tuple[int,int]]): # of output channels, if Tuple, will also specify intermediate # of channels.
            upsampling_mode (str): 'conv' or 'interp', how to do the upsampling
            key (PRNGKeyArray): PRNG key, keyword-only
            kernel_size (Union[int, Sequence[int]], optional): size of convolution kernel. Defaults to 3.
            stride (Union[int, Sequence[int]], optional): convolutional stride. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding to add. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): convolutional dilation. Defaults to 1.
            groups (int, optional): # of groups for convolution. Defaults to 1.
            use_bias (bool, optional): whether to use a bias term or not. Defaults to False.
            padding_mode (str, optional): values used for padding. Defaults to "ZEROS".
            dtype (_type_, optional): datatype of weights for this block. Defaults to None.
            activation (str, optional): activation function to use after convolutions. Defaults to "leaky_relu".
            dropout_prob (float, optional): probability of dropout in the convolution block. Defaults to 0.3.
        """
        key1, key2 = jax.random.split(key, 2)
        self.block = ConvBlock(
            num_spatial_dims,
            False,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            activation,
            dropout_prob,
            key=key1,
        )
        if upsampling_mode == "conv":
            self.upsample = eqx.nn.ConvTranspose(
                num_spatial_dims,
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
                key=key2,
            )
        else:
            self.upsample = Partial(
                upsample_2x,
                num_spatial_dims=num_spatial_dims,
                method=upsampling_mode,
            )

    def __call__(self, x: Array, enc: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the decoder block.

        Takes in the current array being fed through the network and the "encoding" that comes from a skip connection made from the encoder at the equivalent resolution, combines them (channel-wise), and then does convolutions.

        Args:
            x (Array): input array
            enc (Array): encoded array from "across" the UNet
            key (PRNGKeyArray): PRNG key

        Returns:
            Array
        """
        y = self.upsample(x)
        return self.block(jax.numpy.concatenate([y, enc], axis=0), key)
