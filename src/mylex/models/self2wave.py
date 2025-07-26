import os
from collections.abc import Callable
from typing import List, Sequence, Tuple, Union

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy
import optax
import tifffile
from calcite.wavelet import morlet
from jax.tree_util import Partial
from jaxtyping import Array, PRNGKeyArray, PyTree
from matplotlib.pyplot import imread as plt_imread
from parx.pool import PartialMaxPool
from tqdm.auto import trange

from .._types import Scalar
from ..bernoulli import BernoulliMaskMaker
from ..conv import WaveletConvBlock, WaveletPConvBlock
from ..loss import loss_s2s
from ..noise import AdditiveWhiteGaussianNoise
from ..up import UpBlock
from ..util import normalize_0_to_1


class WNet(eqx.Module):
    """WNet architecture that uses convolution with wavelet filter banks in the encoder."""

    encoder_layers: List[WaveletConvBlock | WaveletPConvBlock]
    decoder_layers: List[UpBlock]
    maxpool_layer: PartialMaxPool | eqx.nn.Pool
    output_dropout: eqx.nn.Dropout
    output_conv: eqx.nn.Conv
    output_activation: Callable[[Array], Array]
    partial_convs: bool = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        filter_bank: Array,
        trainable_fbank: bool,
        enc_channels: Sequence[int],
        dec_channels: Sequence[Union[int, Tuple[int, int]]],
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 1,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        upsampling_mode: str = "linear",
        activation: str = "relu",
        output_activation: str = "sigmoid",
        dropout_prob: float = 0.0,
        partial_convs: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the network.

        Args:
            num_spatial_dims (int): number of spatial dimensions
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            filter_bank (Array): wavelet filter bank to use at each layer of encoder.
            trainable_fbank (bool): whether the weights of the filter bank is trainable.
            key (PRNGKeyArray): a `jax.random.PRNGKey` used to provide randomness for parameter initialization. (Keyword only argument)
            enc_channels (Sequence[int], optional): list of number of channels output at each layer of the encoder. Defaults to [48,]*6.
            dec_channels (Sequence[Union[int,Tuple[int,int]]], optional): list of number of channels output at each layer of the decoder. Tuples can be specified so that the intermediate channel at the block can be specified. Defaults to [96, 96, 96, 96, (64, 32)].
            kernel_size (int | Sequence[int], optional): size of convolutional kernels. Defaults to 3.
            stride (int | Sequence[int], optional): convolution stride. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): padding of the convolution. Defaults to 0.
            dilation (int | Sequence[int], optional): dilation of the convolution. Defaults to 1.
            groups (int, optional): number of input channel groups. Defaults to 1.
            use_bias (bool, optional): whether to add a bis on after each convolution. Defaults to False.
            padding_mode (str, optional): string to specify padding values. Defaults to "ZEROS".
            dtype (_type_, optional): dtype to use for the weight and bias in this layer. Defaults to None.
            upsampling_mode (str, optional): how upsampling is handled by the decoder arm. Can be one of the modes for `jax.image.resize` or 'conv'. Defaults to "linear".
            activation (str, optional): activation to use after each convolution. Defaults to 'leaky_relu'.
            output_activation (str, optional): activation of output. Defaults to 'sigmoid'.
            dropout_prob (float, optional): dropout probability. Defaults to 0.0.
            partial_convs (bool, optional): use partial convolutions in encoder. Defaults to True.
        """
        self.partial_convs = partial_convs
        n_enc_layers = len(dec_channels) + 1
        # setup the encoding pathway
        enc_keys = jax.random.split(key, n_enc_layers + 1)
        self.encoder_layers = list()
        for in_chan, out_chan, ekey in zip(
            [in_channels] + list(enc_channels[:-1]), enc_channels, enc_keys[:-1]
        ):
            single_conv = in_chan == in_channels
            self.encoder_layers.append(
                WaveletPConvBlock(
                    single_conv,
                    in_chan,
                    out_chan,
                    filter_bank,
                    None,
                    trainable_fbank,
                    trainable_fbank,
                    use_bias,
                    padding_mode,
                    activation,
                    0.0,
                    True,
                    key=ekey,
                )
                if partial_convs
                else WaveletConvBlock(
                    single_conv,
                    in_chan,
                    out_chan,
                    filter_bank,
                    None,
                    trainable_fbank,
                    trainable_fbank,
                    use_bias,
                    padding_mode,
                    activation,
                    0.0,
                    key=ekey,
                )
            )
        self.maxpool_layer = (
            PartialMaxPool(num_spatial_dims, 2, 2)
            if partial_convs
            else eqx.nn.Pool(-jnp.inf, jax.lax.max, num_spatial_dims, 2, 2)
        )
        # setup the decoding pathway
        concat_chan = reversed([in_channels] + list(enc_channels[:-2]))
        dec_in_chan = [
            c + (p if isinstance(p, int) else p[0])
            for c, p in zip(
                concat_chan, [enc_channels[-1]] + list(dec_channels)
            )
        ]
        dec_keys = jax.random.split(enc_keys[-1], len(dec_channels) + 1)
        self.decoder_layers = list()
        for in_chan, out_chan, dkey in zip(dec_in_chan, dec_channels, dec_keys):
            self.decoder_layers.append(
                UpBlock(
                    num_spatial_dims,
                    in_chan,
                    out_chan,
                    upsampling_mode,
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
                    key=dkey,
                )
            )
        # setup the output convolution & activations
        self.output_dropout = eqx.nn.Dropout(p=dropout_prob)
        if isinstance(dec_channels[-1], int):
            last_dec = dec_channels[-1]
        else:
            last_dec = dec_channels[-1][1]
        self.output_conv = eqx.nn.Conv(
            num_spatial_dims,
            last_dec,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=dec_keys[-1],
        )
        if output_activation == "sigmoid":
            self.output_activation = jax.nn.sigmoid
        elif output_activation == "relu":
            self.output_activation = jax.nn.relu
        elif output_activation == "leaky_relu":
            self.output_activation = Partial(
                jax.nn.leaky_relu, negative_slope=0.1
            )
        else:
            raise ValueError("invalid output_activation")

    def _call_partial(self, x: Array, mask: Array, key: PRNGKeyArray) -> Array:
        # encoder layers
        encoder_depth = len(self.encoder_layers)
        intermediate_encodings = list()
        enc_keys = jax.random.split(key, encoder_depth + 1)
        for idx, encoder_layer in enumerate(self.encoder_layers):
            if idx < encoder_depth - 1:
                intermediate_encodings.append(x)
            x, mask = encoder_layer(x, mask, enc_keys[idx])  # type: ignore
            if idx < encoder_depth - 1:
                x, mask = self.maxpool_layer(x, mask)  # type: ignore
        # decoder
        dec_keys = jax.random.split(enc_keys[-1], len(self.decoder_layers))
        for decoder_layer, dkey in zip(self.decoder_layers, dec_keys):
            x = decoder_layer(x, intermediate_encodings.pop(-1), dkey)
        # output layers
        _, out_key = jax.random.split(dec_keys[-1], 2)
        x = self.output_dropout(x, key=out_key)
        return self.output_activation(self.output_conv(x))

    def _call_notpart(self, x: Array, key: PRNGKeyArray) -> Array:
        # encoder layers/encoder
        encoder_depth = len(self.encoder_layers)
        intermediate_encodings = list()
        enc_keys = jax.random.split(key, encoder_depth + 1)
        for idx, encoder_layer in enumerate(self.encoder_layers):
            if idx < encoder_depth - 1:
                intermediate_encodings.append(x)
            x = encoder_layer(x, enc_keys[idx])  # type: ignore
            if idx < encoder_depth - 1:
                x = self.maxpool_layer(x)  # type: ignore
        # decoder layers/decoder
        dec_keys = jax.random.split(enc_keys[-1], len(self.decoder_layers) + 1)
        for decoder_layer, dkey in zip(self.decoder_layers, dec_keys[:-1]):
            x = decoder_layer(x, intermediate_encodings.pop(-1), dkey)
        # output layers
        x = self.output_dropout(x, key=dec_keys[-1])
        return self.output_activation(self.output_conv(x))

    def __call__(self, x: Array, mask: Array, key: PRNGKeyArray) -> Array:
        """Generate predictions for input array and its mask (forward pass).

        Args:
            x (Array): input array
            mask (Array): mask array
            key (PRNGKeyArray): PRNG key

        Returns:
            Array
        """
        if self.partial_convs:
            return self._call_partial(x, mask, key)
        else:
            return self._call_notpart(x, key)


def train(
    model: WNet,
    image: Array,
    optim: optax.GradientTransformation,
    masker: BernoulliMaskMaker,
    steps: int,
    augment_flips: bool,
    verbose: bool,
    key: PRNGKeyArray,
) -> Tuple[WNet, numpy.ndarray]:
    """Train a model to denoise input images, using the method described in Self2Self.

    Args:
        model (WUNet): the model, a WUNet
        image (Array): noisy image, to be trained to denoise.
        optim (optax.GradientTransformation): the optimizer (from `optax`)
        masker (BernoulliMaskMaker): module to do the Bernoulli masking of the input array.
        steps (int): number of training iterations to do
        augment_flips (bool): also do (probabilistic) vertical & horizontal flips to input during training.
        verbose (bool): show progress bar during training, counting down # of steps.
        key (PRNGKeyArray): PRNG key

    Returns:
        WUNet: trained model
    """
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def _loss(
        model: WNet, image: Array, mask: Array, key: PRNGKeyArray
    ) -> Scalar:
        pred = jax.vmap(model)(jnp.multiply(image, mask), mask, key)
        return loss_s2s(image, pred, mask)

    _loss = eqx.filter_jit(_loss)

    @eqx.filter_jit
    def _make_step(
        model: WNet, x: Array, opt_state: PyTree, key: PRNGKeyArray
    ) -> Tuple[WNet, optax.OptState, Scalar]:
        key_mask, key_loss = jax.random.split(key, 2)
        # batch-ify the keys
        key_mask = jax.random.split(key_mask, x.shape[0])
        key_loss = jax.random.split(key_loss, x.shape[0])
        # maybe augment by randomly flipping the image along either/both axes
        if augment_flips:
            do_flip = numpy.random.rand(2) > 0.5
            if do_flip[0]:
                x = x[:, ::-1, :]
            if do_flip[1]:
                x = x[..., ::-1]
        mask = jax.vmap(masker)(x, key=key_mask)
        loss_val, grads = eqx.filter_value_and_grad(_loss)(
            model, x, mask, key_loss
        )
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    losses = numpy.zeros((steps,), dtype=numpy.float32)
    prog_bar = trange(steps, desc="S2W Training") if verbose else range(steps)
    for step_idx in prog_bar:
        _, key = jax.random.split(key)
        model, opt_state, train_loss = _make_step(model, image, opt_state, key)
        losses[step_idx] = train_loss
        if verbose:
            prog_bar.set_postfix({"loss": round(train_loss * 100, 2)})  # type: ignore
    return model, losses


def test(
    model: WNet,
    image: Array,
    masker: BernoulliMaskMaker,
    n_samples: int = 50,
    batched: bool = False,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Generate model prediction by sampling and averaging many masked instances (see [1] for description).

    Args:
        model (WUNet): the (presumably trained) model
        image (Array): image to denoise
        masker (BernoulliMaskMaker): masking module for Bernoulli-sampled masks
        key (PRNGKeyArray): PRNG key
        n_samples (int, optional): number of samples to take and average. Defaults to 50.
        batched (bool, optional): whether to do all the samples in a single, batch-style or loop through and calculate running parameters to save memory. Defaults to False.

    Raises:
        NotImplementedError: batched=False not implemented, yet.

    Returns:
        Array

    References:
        [1] Quan, et al "Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image", CVPR 2020
    """
    key_model, key_mask = jax.random.split(key, 2)
    if batched:
        key_mask = jax.random.split(key_mask, n_samples)
        key_model = jax.random.split(key_model, n_samples)
        image = jax.lax.stop_gradient(image.repeat(n_samples, axis=0))
        mask = jax.lax.stop_gradient(jax.vmap(masker)(image, key=key_mask))
        pred = jax.vmap(model)(image, mask, key_model)
        return jax.numpy.mean(pred, axis=0)
    else:
        image = jax.lax.stop_gradient(image[0])
        preds = []
        for _ in range(n_samples):
            key_mask, key_mask_iter = jax.random.split(key_mask, 2)
            key_model, key_model_iter = jax.random.split(key_model, 2)
            mask = jax.lax.stop_gradient(masker(image, key=key_mask_iter))
            pred = model(image, mask, key_model_iter)
            preds.append(pred)
        return jax.numpy.mean(jax.numpy.stack(preds, axis=0), axis=0)


@click.group()
def command_line_interface() -> int:
    return 0


@command_line_interface.command("train")
@click.option("--img-path", default=None, help="path to image to denoise")
@click.option(
    "--num-iter", default=1, help="# of steps to take during training"
)
@click.option(
    "--learning-rate", default=0.01, help="Learning rate of Adam optimizer"
)
@click.option(
    "--prob-mask",
    default=0.3,
    help="Probability of a pixel being masked at each iteration",
)
@click.option(
    "--num-stds", default=2, help="# of std. devs wide to make filters"
)
@click.option(
    "--num-orientations", default=8, help="# of orientations in the filter bank"
)
@click.option("--trainable-filters", is_flag=True)
@click.option("--mask-indep-channels", is_flag=True)
@click.option("--augment-flips", is_flag=True)
@click.option("--prng-seed", default=1, help="PRNG Seed")
@click.option(
    "--output-path", default=None, help="path to save trained model to"
)
def do_training(
    img_path: str,
    num_iter: int,
    learning_rate: float,
    prob_mask: float,
    num_stds: float,
    num_orientations: int,
    trainable_filters: bool,
    mask_indep_channels: bool,
    augment_flips: bool,
    prng_seed: int,
    output_path: str,
) -> int:
    if img_path is None:
        raise ValueError("must specify an input path to an image")
    # read the image in
    if img_path[-3:] == "png":
        img = numpy.asarray(plt_imread(img_path), dtype=numpy.float32)
    else:
        img = numpy.asarray(tifffile.imread(img_path), dtype=numpy.float32)
    if len(img.shape) == 2:
        img = numpy.expand_dims(img, 0)
        in_chan = 1
    else:
        in_chan = img.shape[0]
    if not len(img.shape) == 3:
        raise ValueError(
            "invalid input, image array must be 2 or 2+1-dimensional"
        )
    img = jax.numpy.expand_dims(jax.numpy.asarray(img), 0)
    img = jax.vmap(normalize_0_to_1, 0, 0)(img)
    # make the filter bank
    fbank = morlet.filter_bank_2d_scikit(
        n_stds=num_stds, n_orientations=num_orientations
    )
    # initialize the model
    key = jax.random.key(prng_seed)
    model_key, train_key = jax.random.split(key, 2)
    model = WNet(
        2,
        in_chan,
        in_chan,
        fbank,
        trainable_filters,
        [24, 48, 48, 48, 48],
        [96, 96, 96, (64, 32)],
        key=model_key,
    )
    # initialize the optimizer & masker
    opt = optax.adam(learning_rate)
    bmask = BernoulliMaskMaker(prob_mask, mask_indep_channels)
    # do the training
    model, losses = train(
        model, img, opt, bmask, num_iter, augment_flips, True, train_key
    )
    # save the model, generate a denoised output
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        numpy.savetxt(
            os.path.join(output_path, "loss.csv"), losses, delimiter=","
        )
        eqx.tree_serialise_leaves(os.path.join(output_path, "model.eqx"), model)
    return 0


@command_line_interface.command("train-awgn")
@click.option("--sigma", default=0.5, help="std. dev of added gaussian noise")
@click.option("--img-path", default=None, help="Path to image to denoise")
@click.option(
    "--num-iter", default=1, help="Number of steps to take during training"
)
@click.option(
    "--learning-rate", default=0.01, help="Learning rate of Adam optimizer"
)
@click.option(
    "--prob-mask",
    default=0.3,
    help="Probability of a pixel being masked at each iteration",
)
@click.option(
    "--num-stds", default=2, help="# of std. devs wide to make filters"
)
@click.option(
    "--num-orientations", default=8, help="# of orientations in the filter bank"
)
@click.option("--trainable-filters", is_flag=True)
@click.option("--mask-indep-channels", is_flag=True)
@click.option("--augment-flips", is_flag=True)
@click.option("--prng-seed", default=1, help="PRNG Seed")
@click.option(
    "--output-path", default=None, help="path to save trained model to"
)
def do_training_awgn(
    sigma: float,
    img_path: str,
    num_iter: int,
    learning_rate: float,
    prob_mask: float,
    num_stds: float,
    num_orientations: int,
    trainable_filters: bool,
    mask_indep_channels: bool,
    augment_flips: bool,
    prng_seed: int,
    output_path: str,
) -> int:
    if img_path is None:
        raise ValueError("must specify an input path to an image")
    # read the image in
    if img_path[-3:] == "png":
        img = numpy.asarray(plt_imread(img_path), dtype=numpy.float32)
    else:
        try:
            img = numpy.asarray(tifffile.imread(img_path), dtype=numpy.float32)
        except Exception:
            raise ValueError("input image must be a PNG or a TIF")
    if len(img.shape) == 2:
        img = numpy.expand_dims(img, 0)
        in_chan = 1
    else:
        in_chan = img.shape[0]
    if not len(img.shape) == 3:
        raise ValueError(
            "invalid input, image array must be 2 or 3-dimensional"
        )
    img = jax.numpy.expand_dims(jax.numpy.asarray(img), 0)
    key = jax.random.key(prng_seed)
    noise_key, model_key, train_key, denoise_key = jax.random.split(key, 4)
    noiser = AdditiveWhiteGaussianNoise(sigma)
    img = noiser(img, noise_key)
    img = jax.vmap(normalize_0_to_1, 0, 0)(img)
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        numpy.save(
            os.path.join(output_path, "noisy_image.npy"), numpy.asarray(img[0])
        )
    # initialize the model
    fbank = morlet.filter_bank_2d_scikit(
        n_stds=num_stds, n_orientations=num_orientations
    )
    fbank = jnp.concatenate([fbank.real, fbank.imag], axis=0)
    model = WNet(
        2,
        in_chan,
        in_chan,
        fbank,
        trainable_filters,
        [32, 48, 48, 48, 48],
        [96, 96, 96, (64, 32)],
        key=model_key,
    )
    # initialize the optimizer & masker
    opt = optax.adam(learning_rate)
    bmask = BernoulliMaskMaker(prob_mask, mask_indep_channels)
    # do the training
    model, losses = train(
        model, img, opt, bmask, num_iter, augment_flips, True, train_key
    )
    # save the model, generate a denoised prediction
    if output_path is not None:
        numpy.savetxt(
            os.path.join(output_path, "loss.csv"), losses, delimiter=","
        )
        eqx.tree_serialise_leaves(os.path.join(output_path, "model.eqx"), model)
        denoised = numpy.asarray(
            test(model, img, bmask, 50, False, key=denoise_key)
        )
        numpy.save(os.path.join(output_path, "denoised_image.npy"), denoised)
    return 0
