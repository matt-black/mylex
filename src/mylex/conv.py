from collections.abc import Callable
from typing import Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, PRNGKeyArray
from parx.conv import PartialConv


class ConvDWS(eqx.Module):
    """A depthwise-separable convolution."""

    dwconv: eqx.nn.Conv  # depthwise
    pwconv: eqx.nn.Conv  # pointwise

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the convolution.

        Args:
            num_spatial_dims (int): number of spatial dimensions in inputs
            in_channels (int): # of input channels
            out_channels (int): # of output channels
            kernel_size (int): size of convolution kernel
            stride (Union[int, Sequence[int]], optional): convolution stride. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): convolution dilation. Defaults to 1.
            use_bias (bool, optional): whether or not to include a bias term in convolutions. Defaults to False.
            padding_mode (str, optional): values used for padding. Defaults to "ZEROS".
            dtype (optional): datatype used for weights. Defaults to None.
            key (PRNGKeyArray): PRNG key, keyword-only
        """
        dwkey, pwkey = jax.random.split(key, 2)
        self.dwconv = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            use_bias=use_bias,
            padding_mode=padding_mode,
            dtype=dtype,
            key=dwkey,
        )
        self.pwconv = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            1,
            1,
            0,
            1,
            1,
            use_bias=False,
            padding_mode="ZEROS",
            key=pwkey,
        )

    def __call__(self, x: Array) -> Array:
        return self.pwconv(self.dwconv(x))


class ConvBlock(eqx.Module):
    """A block of convolutions with dropout."""

    conv1: eqx.nn.Conv | ConvDWS
    dropout: eqx.nn.Dropout
    conv2: eqx.nn.Conv | ConvDWS | None
    activation: Callable[[Array], Array]

    def __init__(
        self,
        num_spatial_dims: int,
        single_conv: bool,
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
        depthwise_separable: bool = False,
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
            depthwise_separable (bool, optional): use depth-wise separable convolutions. Defaults to False.

        Raises:
            ValueError: if invalid activation function specified
        """
        key1, key2 = jax.random.split(key, 2)
        if isinstance(out_channels, int):
            out_channels = (out_channels, out_channels)
        if depthwise_separable:
            self.conv1 = ConvDWS(
                num_spatial_dims,
                in_channels,
                out_channels[0],
                kernel_size,
                stride,
                padding,
                dilation,
                use_bias=use_bias,
                padding_mode=padding_mode,
                dtype=dtype,
                key=key1,
            )
            if single_conv:
                self.conv2 = None
            else:
                self.conv2 = ConvDWS(
                    num_spatial_dims,
                    out_channels[0],
                    out_channels[1],
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    use_bias=use_bias,
                    padding_mode=padding_mode,
                    dtype=dtype,
                    key=key1,
                )
        else:
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
            if single_conv:
                self.conv2 = None
            else:
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
        if dropout_prob == 0:
            self.dropout = eqx.nn.Dropout(inference=True)
        else:
            self.dropout = eqx.nn.Dropout(p=dropout_prob)

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


class WaveletConv(eqx.Module):
    """Convolution, similar to a depthwise-separable convolution, using a (possibly fixed) filter bank of wavelets."""

    filter_bank: Array
    num_spatial_dims: int = eqx.field(static=True)
    output_conv: eqx.nn.Conv

    def __init__(
        self,
        filter_bank: Array,
        in_channels: int,
        out_channels: int,
        trainable: bool,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        *,
        key: PRNGKeyArray,
    ):
        _, ockey = jax.random.split(key, 2)
        # input filter bank should have size [N,...] where ... are spatial dimensions and N is the number of filters in the filter bank
        # add singleton dimension at axis=1 b/c JAX wants the convolution kernel to be in the form OIHW, and we want each kernel to be applied to each channel
        self.filter_bank = (
            jax.lax.stop_gradient(jnp.expand_dims(filter_bank, 1))
            if trainable
            else jnp.expand_dims(filter_bank, 1)
        )
        num_spatial_dims = len(filter_bank.shape) - 1
        self.num_spatial_dims = num_spatial_dims
        num_fbank_chans = filter_bank.shape[0]
        num_interm_chans = num_fbank_chans * in_channels
        self.output_conv = eqx.nn.Conv(
            num_spatial_dims,
            num_interm_chans,
            out_channels,
            1,
            1,
            padding="same",
            use_bias=use_bias,
            padding_mode=padding_mode,
            key=ockey,
        )

    def _conv(self, x: Array) -> Array:
        """Convolves the input, `x`, with the filter bank and return the output."""
        return jax.lax.conv(
            x[None, None, ...],
            self.filter_bank,
            window_strides=[
                1,
            ]
            * self.num_spatial_dims,
            padding="same",
        )[0]

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        y = jnp.reshape(jax.vmap(self._conv, 0, 0)(x), [-1] + list(x.shape[1:]))
        return self.output_conv(y, key=key)


class WaveletPConv(eqx.Module):
    """Partial convolution, similar to a depthwise-separable convolution, using a (possibly fixed) filter bank of wavelets."""

    pconv: PartialConv
    num_spatial_dims: int = eqx.field(static=True)
    return_mask: bool = eqx.field(static=True)
    output_conv: PartialConv

    def __init__(
        self,
        filter_bank: Array,
        in_channels: int,
        out_channels: int,
        trainable_fbank: bool,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        return_mask: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the module.

        Args:
            filter_bank (Array): the wavelet filter bank.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            trainable_fbank (bool): whether the filter bank can be updated during training.
            key (PRNGKeyArray): PRNG key array.
            use_bias (bool, optional): use a bias term with the convolution. Defaults to False.
            padding_mode (str, optional): values used for padding. Defaults to "ZEROS".
            return_mask (bool, optional): whether to return the updated mask along with the new array. Defaults to False.
        """
        pkey, ockey = jax.random.split(key, 2)
        num_spatial_dims = len(filter_bank.shape) - 1
        self.num_spatial_dims = num_spatial_dims
        self.return_mask = return_mask
        num_filts = filter_bank.shape[0]
        ker_size = filter_bank.shape[1:]
        interm_chans = num_filts * in_channels
        self.pconv = PartialConv(
            num_spatial_dims,
            in_channels,
            interm_chans,
            ker_size,
            1,
            "same",
            1,
            1,
            use_bias,
            padding_mode,
            None,
            True,
            False,
            # kwargs
            weight=filter_bank[:, None, ...],
            fixed=(not trainable_fbank),
            key=pkey,
        )
        self.output_conv = PartialConv(
            num_spatial_dims,
            interm_chans,
            out_channels,
            1,
            1,
            "same",
            1,
            1,
            use_bias,
            padding_mode,
            None,
            return_mask=return_mask,
            fft_conv=False,
            key=ockey,
        )

    def __call__(self, x: Array, mask: Array) -> Array | Tuple[Array, Array]:
        """Forward pass through the module.

        This applies all the filters in the filter bank to all the channels of the input.

        Args:
            x (Array): input array
            mask (Array): mask array

        Returns:
            Array | Tuple[Array, Array]: output array, and possibly the mask.
        """
        # vmap over pconv does the vmap over channels so that all filters in the filter bank are applied to all of the input channels
        y, mask = jax.vmap(self.pconv, (0, 0), (0, 0))(
            jnp.expand_dims(x, 1),
            jnp.expand_dims(mask, 1),
        )
        reshp = [
            -1,
        ] + list(y.shape[2:])
        mask = mask.reshape(reshp)
        z, mask = self.output_conv(y.reshape(reshp), mask)
        if self.return_mask:
            return z, mask
        else:
            return z


class ScatteringConv(eqx.Module):
    """A convolution layer used as part of a learnable scattering transform.

    This layer takes in a multi-channel input and applies all of the filters in its filter bank to each of the channels. The output of this is then (# input channels x # filters). This is then convolved with a 1x1 conv kernel (pointwise convolution) to the desired number of output channels. Note that this is similar to a depthwise-separable convolution.
    """

    filter_bank: Array
    num_spatial_dims: int = eqx.field(static=True)
    output_conv: eqx.nn.Conv
    activation: Callable[[Array], Array]

    def __init__(
        self,
        filter_bank: Array,
        in_channels: int,
        out_channels: int,
        trainable_fbank: bool,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        activation: str | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the module.

        Args:
            filter_bank (Array): the wavelet filter bank.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            trainable_fbank (bool): whether the filter bank can be updated during training.
            key (PRNGKeyArray): PRNG key array.
            use_bias (bool, optional): use a bias term with the convolution. Defaults to False.
            padding_mode (str, optional): values used for padding. Defaults to "ZEROS".
            activation (str, optional): the activation applied after the filter bank convolution, but before the pointwise (1x1) convolution. If None, no activation is applied. Defaults to None.
        """
        _, ockey = jax.random.split(key, 2)
        # input filter bank should have size [N,...] where ... are spatial dimensions and N is the number of filters in the filter bank
        # add singleton dimension at axis=1 b/c JAX wants the convolution kernel to be in the form OIHW, and we want each kernel to be applied to each channel
        self.filter_bank = (
            jax.lax.stop_gradient(jnp.expand_dims(filter_bank, 1))
            if trainable_fbank
            else jnp.expand_dims(filter_bank, 1)
        )
        num_spatial_dims = len(filter_bank.shape) - 1
        self.num_spatial_dims = num_spatial_dims
        num_fbank_chans = filter_bank.shape[0]
        num_interm_chans = num_fbank_chans * in_channels
        self.output_conv = eqx.nn.Conv(
            num_spatial_dims,
            num_interm_chans,
            out_channels,
            1,
            1,
            padding="same",
            use_bias=use_bias,
            padding_mode=padding_mode,
            key=ockey,
        )
        if activation is None:
            self.activation = jax.nn.identity
        else:
            if activation == "relu":
                self.activation = jax.nn.relu
            elif activation == "leaky_relu":
                self.activation = Partial(jax.nn.leaky_relu, negative_slope=0.1)
            elif activation == "silu":
                self.activation = jax.nn.silu
            else:
                raise ValueError("invalid activation function")

    def _conv(self, x: Array) -> Array:
        """Convolves the input, `x`, with the filter bank and return the output."""
        return jax.lax.conv(
            x[None, None, ...],
            self.filter_bank,
            window_strides=[
                1,
            ]
            * self.num_spatial_dims,
            padding="same",
        )[0]

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass through the module.

        This applies all the filters in the filter bank to all the channels of the input.

        Args:
            x (Array): input array
            key (PRNGKeyArray): PRNG key array.

        Returns:
            Array: output array
        """
        y = jnp.reshape(jax.vmap(self._conv, 0, 0)(x), [-1] + list(x.shape[1:]))
        return self.output_conv(self.activation(y), key=key)


class ScatteringPConv(eqx.Module):
    """A partial convolution layer used as part of a learnable scattering transform.

    This layer takes in a multi-channel input and applies all of the filters in its filter bank to each of the channels. The output of this is then (# input channels x # filters). This is then convolved with a 1x1 conv kernel (pointwise convolution) to the desired number of output channels. Note that this is similar to a depthwise-separable convolution.
    """

    pconv: PartialConv
    num_spatial_dims: int = eqx.field(static=True)
    return_mask: bool = eqx.field(static=True)
    output_conv: PartialConv
    activation: Callable[[Array], Array]

    def __init__(
        self,
        filter_bank: Array,
        in_channels: int,
        out_channels: int,
        trainable_fbank: bool,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        return_mask: bool = False,
        activation: str | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the module.

        Args:
            filter_bank (Array): the wavelet filter bank.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            trainable_fbank (bool): whether the filter bank can be updated during training.
            key (PRNGKeyArray): PRNG key array.
            use_bias (bool, optional): use a bias term with the convolution. Defaults to False.
            padding_mode (str, optional): values used for padding. Defaults to "ZEROS".
            return_mask (bool, optional): whether to return the updated mask along with the new array. Defaults to False.
        """
        pkey, ockey = jax.random.split(key, 2)
        num_spatial_dims = len(filter_bank.shape) - 1
        self.num_spatial_dims = num_spatial_dims
        self.return_mask = return_mask
        num_filts = filter_bank.shape[0]
        ker_size = filter_bank.shape[1:]
        interm_chans = num_filts * in_channels
        self.pconv = PartialConv(
            num_spatial_dims,
            in_channels,
            interm_chans,
            ker_size,
            1,
            "same",
            1,
            1,
            use_bias,
            padding_mode,
            None,
            True,
            False,
            # kwargs
            weight=filter_bank[:, None, ...],
            fixed=(not trainable_fbank),
            key=pkey,
        )
        self.output_conv = PartialConv(
            num_spatial_dims,
            interm_chans,
            out_channels,
            1,
            1,
            "same",
            1,
            1,
            use_bias,
            padding_mode,
            None,
            return_mask=return_mask,
            fft_conv=False,
            key=ockey,
        )
        if activation is None:
            self.activation = jax.nn.identity
        else:
            if activation == "relu":
                self.activation = jax.nn.relu
            elif activation == "leaky_relu":
                self.activation = Partial(jax.nn.leaky_relu, negative_slope=0.1)
            elif activation == "silu":
                self.activation = jax.nn.silu
            else:
                raise ValueError("invalid activation function")

    def __call__(self, x: Array, mask: Array) -> Array | Tuple[Array, Array]:
        """Forward pass through the module.

        This applies all the filters in the filter bank to all the channels of the input.

        Args:
            x (Array): input array
            mask (Array): mask array

        Returns:
            Array | Tuple[Array, Array]: output array, and possibly the mask.
        """
        # vmap over pconv does the vmap over channels so that all filters in the filter bank are applied to all of the input channels
        y, mask = jax.vmap(self.pconv, (0, 0), (0, 0))(
            jnp.expand_dims(x, 1),
            jnp.expand_dims(mask, 1),
        )
        reshp = [
            -1,
        ] + list(y.shape[2:])
        mask = mask.reshape(reshp)
        z, mask = self.output_conv(self.activation(y.reshape(reshp)), mask)
        if self.return_mask:
            return z, mask
        else:
            return z


class WaveletConvBlock(eqx.Module):
    """A block of convolutions using a wavelet filter bank, with dropout."""

    conv1: WaveletConv
    dropout: eqx.nn.Dropout
    conv2: WaveletConv | None
    activation: Callable[[Array], Array]

    def __init__(
        self,
        single_conv: bool,
        in_channels: int,
        out_channels: Union[int, Tuple[int, int]],
        filter_bank1: Array,
        filter_bank2: Array | None,
        trainable_fbank1: bool,
        trainable_fbank2: bool,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        activation: str | None = None,
        dropout_prob: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the module.

        Args:
            single_conv (bool): if True, only a single convolution is included in the block.
            in_channels (int): number of input channels.
            out_channels (Union[int, Tuple[int,int]]): number of output channels. If a tuple is used, then its (# intermediate channels, # output channels).
            filter_bank1 (Array): filter bank in first convolution.
            filter_bank2 (Array | None): filter bank in second convolution. If None, the filter bank from the first layer is reused.
            trainable_fbank1 (bool): whether the first filter bank's weights are trainable.
            trainable_fbank2 (bool): whether the second filter bank's weights are trainable.
            key (PRNGKeyArray): PRNG key array.
            use_bias (bool, optional): use a bias with convolutions. Defaults to False.
            padding_mode (str, optional): value to use for padding. Defaults to "ZEROS".
            activation (str, optional): activation function to use after convolutions. Defaults to "relu".
            dropout_prob (float, optional): probability of dropout between convolutions. Defaults to 0.0.

        Raises:
            ValueError: if invalid activation is specified.
        """
        key1, key2 = jax.random.split(key, 2)
        if isinstance(out_channels, int):
            out_channels = (out_channels, out_channels)
        self.conv1 = WaveletConv(
            filter_bank1,
            in_channels,
            out_channels[0],
            trainable_fbank1,
            use_bias,
            padding_mode,
            key=key1,
        )
        if single_conv:
            self.conv2 = None
        else:
            self.conv2 = WaveletConv(
                filter_bank2 if filter_bank2 is not None else filter_bank1,
                out_channels[0],
                out_channels[1],
                trainable_fbank2,
                use_bias,
                padding_mode,
                key=key2,
            )
        if dropout_prob == 0:
            self.dropout = eqx.nn.Dropout(inference=True)
        else:
            self.dropout = eqx.nn.Dropout(p=dropout_prob)

        if activation is None:
            self.activation = jax.nn.identity
        else:
            if activation == "relu":
                self.activation = jax.nn.relu
            elif activation == "leaky_relu":
                self.activation = Partial(jax.nn.leaky_relu, negative_slope=0.1)
            elif activation == "silu":
                self.activation = jax.nn.silu
            else:
                raise ValueError("invalid activation function")

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass through the module.

        Args:
            x (Array): input array.
            key (PRNGKeyArray): PRNG key array.

        Returns:
            Array: output array.
        """
        k1, kd, k2 = jax.random.split(key, 3)
        y = self.activation(self.conv1(x, k1))
        z = self.dropout(y, key=kd)
        if self.conv2 is None:
            return z
        else:
            return self.activation(self.conv2(z, key=k2))


class WaveletPConvBlock(eqx.Module):
    """A block of partial convolutions using a wavelet filter bank, with dropout."""

    conv1: WaveletPConv
    dropout: eqx.nn.Dropout
    conv2: WaveletPConv | None
    activation: Callable[[Array], Array]
    return_mask: bool = eqx.field(static=True)

    def __init__(
        self,
        single_conv: bool,
        in_channels: int,
        out_channels: Union[int, Tuple[int, int]],
        filter_bank1: Array,
        filter_bank2: Array | None,
        trainable_fbank1: bool,
        trainable_fbank2: bool,
        use_bias: bool,
        padding_mode: str = "ZEROS",
        activation: str = "relu",
        dropout_prob: float = 0.0,
        return_mask: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the module.

        Args:
            single_conv (bool): if True, only a single convolution is included in the block.
            in_channels (int): number of input channels.
            out_channels (Union[int, Tuple[int,int]]): number of output channels. If a tuple is used, then its (# intermediate channels, # output channels).
            filter_bank1 (Array): filter bank in first convolution.
            filter_bank2 (Array | None): filter bank in second convolution. If None, the filter bank from the first layer is reused.
            trainable_fbank1 (bool): whether the first filter bank's weights are trainable.
            trainable_fbank2 (bool): whether the second filter bank's weights are trainable.
            key (PRNGKeyArray): PRNG key array.
            use_bias (bool, optional): use a bias with convolutions. Defaults to False.
            padding_mode (str, optional): value to use for padding. Defaults to "ZEROS".
            activation (str, optional): activation function to use after convolutions. Defaults to "relu".
            dropout_prob (float, optional): probability of dropout between convolutions. Defaults to 0.0.
            return_mask (bool, optional): whether to return the mask along with the updated array. Defaults to True.

        Raises:
            ValueError: if invalid activation is specified.
        """
        key1, key2 = jax.random.split(key, 2)
        self.return_mask = return_mask
        if isinstance(out_channels, int):
            out_channels = (out_channels, out_channels)
        self.conv1 = WaveletPConv(
            filter_bank1,
            in_channels,
            out_channels[0],
            trainable_fbank1,
            use_bias,
            padding_mode,
            return_mask=True,
            key=key1,
        )
        if single_conv:
            self.conv2 = None
        else:
            self.conv2 = WaveletPConv(
                (filter_bank1 if filter_bank2 is None else filter_bank2),
                out_channels[0],
                out_channels[1],
                trainable_fbank2,
                use_bias,
                padding_mode,
                return_mask=True,
                key=key2,
            )
        if dropout_prob == 0:
            self.dropout = eqx.nn.Dropout(inference=True)
        else:
            self.dropout = eqx.nn.Dropout(p=dropout_prob)

        if activation == "leaky_relu":
            self.activation = Partial(jax.nn.leaky_relu, negative_slope=0.1)
        elif activation == "relu":
            self.activation = jax.nn.relu
        else:
            raise ValueError("only ReLU and Leaky ReLU are valid")

    def __call__(
        self, x: Array, mask: Array, key: PRNGKeyArray
    ) -> Array | Tuple[Array, Array]:
        """Forward pass through the module.

        Args:
            x (Array): input array.
            mask (Array): mask array.
            key (PRNGKeyArray): PRNG key array.

        Returns:
            Array | Tuple[Array, Array]: output array, possibly the mask.
        """
        x1, mask = self.conv1(x, mask)
        y = self.activation(x1)
        z = self.dropout(y, key=key)
        if self.conv2 is None:
            return (z, mask) if self.return_mask else z
        else:
            a, mask = self.conv2(z, mask)
            return (a, mask) if self.return_mask else a
