from collections.abc import Callable
from typing import Sequence, Tuple, Union

import equinox as eqx
import jax
from jax.tree_util import Partial
from jaxtyping import Array, PRNGKeyArray


class ConvBlock(eqx.Module):
    """A block of convolutions with dropout."""

    conv1: eqx.nn.Conv
    dropout: eqx.nn.Dropout
    conv2: eqx.nn.Conv
    activation: Callable[[Array], Array]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: Union[int, Tuple[int, int]],
        kernel_size: int | Sequence[int],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        activation: str = "leaky_relu",
        dropout_prob: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the block of convolutions.

        Args:
            num_spatial_dims (int): number of spatial dimensions in inputs
            in_channels (int): # of input channels
            out_channels (Union[int, Tuple[int, int]]): # of output channels, if a Tuple will specify the # of intermediate channels in the block.
            kernel_size (int): size of convolution kernel
            key (PRNGKeyArray): PRNG key, keyword-only
            stride (Union[int, Sequence[int]], optional): convolution stride. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): convolution dilation. Defaults to 1.
            groups (int, optional): groups used in convolution. Defaults to 1.
            use_bias (bool, optional): whether or not to include a bias term in convolutions. Defaults to False.
            padding_mode (str, optional): values used for padding. Defaults to "ZEROS".
            dtype (_type_, optional): datatype used for weights. Defaults to None.
            activation (str, optional): the activation function to use after convolutions. Defaults to "leaky_relu".
            dropout_prob (float, optional): probability of dropout between the convolutions. Defaults to 0.0.

        Raises:
            ValueError: if invalid activation function specified
        """
        key1, key2 = jax.random.split(key, 2)
        if isinstance(out_channels, int):
            out_channels = (out_channels, out_channels)
        self.conv1 = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels[0],
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=key1,
        )
        if dropout_prob == 0:
            self.dropout = eqx.nn.Dropout(inference=True)
        else:
            self.dropout = eqx.nn.Dropout(p=dropout_prob)
        self.conv2 = eqx.nn.Conv(
            num_spatial_dims,
            out_channels[0],
            out_channels[1],
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=key2,
        )
        if activation == "leaky_relu":
            self.activation = Partial(jax.nn.leaky_relu, negative_slope=0.1)
        elif activation == "relu":
            self.activation = jax.nn.relu
        else:
            raise ValueError("only ReLU and Leaky ReLU are valid")

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass through the block.

        Args:
            x (Array): input array
            key (PRNGKeyArray): PRNG key

        Returns:
            Array
        """
        y = self.activation(self.conv1(x))
        z = self.dropout(y, key=key)
        return z if self.conv2 is None else self.activation(self.conv2(z))
