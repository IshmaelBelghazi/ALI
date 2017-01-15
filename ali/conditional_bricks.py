""" Conditional ALI related Bricks"""

from theano import tensor
from theano import (function, )

from blocks.bricks.base import Brick, application, lazy
from blocks.bricks import LeakyRectifier, Logistic
from blocks.bricks import (Linear, Sequence, )
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence, ConvolutionalSequence, )
from blocks.bricks.interfaces import Initializable, Random

from blocks.initialization import IsotropicGaussian, Constant

from blocks.select import Selector

from ali.bricks import ConvMaxout
from ali.utils import get_log_odds, conv_brick, conv_transpose_brick, bn_brick


class Embedder(Initializable):
    """
    Linear Embedding Brick
    Parameters
    ----------
    dim_in: :class:`int`
        Dimensionality of the input
    dim_out: :class:`int`
        Dimensionality of the output
    output_type: :class:`str`
        fc for fully connected. conv for convolutional
    """

    def __init__(self, dim_in, dim_out, output_type='fc', **kwargs):

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.output_type = output_type
        self.linear = Linear(dim_in, dim_out, name='embed_layer')
        children = [self.linear]
        kwargs.setdefault('children', []).extend(children)
        super(Embedder, self).__init__(**kwargs)

    @application(inputs=['y'], outputs=['outputs'])
    def apply(self, y):
        embedding = self.linear.apply(y)
        if self.output_type == 'fc':
            return embedding
        if self.output_type == 'conv':
            return embedding.reshape((-1, embedding.shape[-1], 1, 1))

    def get_dim(self, name):
        if self.output_type == 'fc':
            return self.linear.get_dim(name)
        if self.output_type == 'conv':
            return (self.linear.get_dim(name), 1, 1)


class EncoderMapping(Initializable):
    """
    Parameters
    ----------
    layers: :class:`list`
        list of bricks
    num_channels: :class: `int`
           Number of input channels
    image_size: :class:`tuple`
        Image size
    n_emb: :class:`int`
        Dimensionality of the embedding
    use_bias: :class:`bool`
        self explanatory
    """
    def __init__(self, layers, num_channels, image_size, n_emb, use_bias=False, **kwargs):
        self.layers = layers
        self.num_channels = num_channels
        self.image_size = image_size

        self.pre_encoder = ConvolutionalSequence(layers=layers[:-1],
                                                 num_channels=num_channels,
                                                 image_size=image_size,
                                                 use_bias=use_bias,
                                                 name='encoder_conv_mapping')
        self.pre_encoder.allocate()
        n_channels = n_emb + self.pre_encoder.get_dim('output')[0]
        self.post_encoder = ConvolutionalSequence(layers=[layers[-1]],
                                                  num_channels=n_channels,
                                                  image_size=(1, 1),
                                                  use_bias=use_bias)
        children = [self.pre_encoder, self.post_encoder]
        kwargs.setdefault('children', []).extend(children)
        super(EncoderMapping, self).__init__(**kwargs)

    @application(inputs=['x', 'y'], outputs=['output'])
    def apply(self, x, y):
        "Returns mu and logsigma"
        # Getting emebdding
        pre_z = self.pre_encoder.apply(x)
        # Concatenating
        pre_z_embed_y = tensor.concatenate([pre_z, y], axis=1)
        # propagating through last layer
        return self.post_encoder.apply(pre_z_embed_y)


class Decoder(Initializable):
    def __init__(self, layers, num_channels, image_size, use_bias=False, **kwargs):
        self.layers = layers
        self.num_channels = num_channels
        self.image_size = image_size

        self.mapping = ConvolutionalSequence(layers=layers,
                                             num_channels=num_channels,
                                             image_size=image_size,
                                             use_bias=use_bias,
                                             name='decoder_mapping')
        children = [self.mapping]
        kwargs.setdefault('children', []).extend(children)
        super(Decoder, self).__init__(**kwargs)

    @application(inputs=['z', 'y'], outputs=['outputs'])
    def apply(self, z, y, application_call):
        # Concatenating conditional data with inputs
        z_y = tensor.concatenate([z, y], axis=1)
        return self.mapping.apply(z_y)


class GaussianConditional(Initializable, Random):
    def __init__(self, mapping, **kwargs):
        self.mapping = mapping
        super(GaussianConditional, self).__init__(**kwargs)
        self.children.extend([mapping])
    @property
    def _nlat(self):
        # if isinstance(self.mapping, ConvolutionalSequence):
        #     return self.get_dim('output')[0]
        # else:
        #     return self.get_dim('output')
        return self.mapping.children[-1].get_dim('output')[0] // 2

    def get_dim(self, name):
        if isinstance(self.mapping, ConvolutionalSequence):
            dim = self.mapping.get_dim(name)
            if name == 'output':
                return (dim[0] // 2) + dim[1:]
            else:
                return dim
        else:
            if name == 'output':
                return self.mapping.output_dim // 2
            elif name == 'input_':
                return self.mapping.input_dim
            else:
                return self.mapping.get_dim(name)
    @application(inputs=['x', 'y'], outputs=['output'])
    def apply(self, x, y, application_call):
        params = self.mapping.apply(x, y)
        mu, log_sigma = params[:, :self._nlat], params[:, self._nlat:]
        sigma = tensor.exp(log_sigma)
        epsilon = self.theano_rng.normal(size=mu.shape)
        return mu + sigma * epsilon


class XZYJointDiscriminator(Initializable):
    """Three-way discriminator.

    Parameters
    ----------
    x_discriminator : :class:`blocks.bricks.Brick`
        Part of the discriminator taking :math:`x` as input. Its
        output will be concatenated with ``z_discriminator``'s output
        and fed to ``joint_discriminator``.
    z_discriminator : :class:`blocks.bricks.Brick`
        Part of the discriminator taking :math:`z` as input. Its
        output will be concatenated with ``x_discriminator``'s output
        and fed to ``joint_discriminator``.
    joint_discriminator : :class:`blocks.bricks.Brick`
        Part of the discriminator taking the concatenation of
        ``x_discriminator``'s and output ``z_discriminator``'s output
        as input and computing :math:`D(x, z)`.

    """
    def __init__(self, x_discriminator, z_discriminator, joint_discriminator,
                 **kwargs):
        self.x_discriminator = x_discriminator
        self.z_discriminator = z_discriminator
        self.joint_discriminator = joint_discriminator

        super(XZYJointDiscriminator, self).__init__(**kwargs)
        self.children.extend([self.x_discriminator, self.z_discriminator,
                              self.joint_discriminator])

    @application(inputs=['x', 'z', 'y'], outputs=['output'])
    def apply(self, x, z, y):
        # NOTE: the unbroadcasts act as a workaround for a weird broadcasting
        # bug when applying dropout
        input_ = tensor.unbroadcast(
            tensor.concatenate(
                [self.x_discriminator.apply(x), self.z_discriminator.apply(z), y],
                axis=1),
            *range(x.ndim))
        return self.joint_discriminator.apply(input_)


class ConditionalALI(Initializable, Random):
    """Adversarial learned inference brick.

    Parameters
    ----------
    encoder : :class:`blocks.bricks.Brick`
        Encoder network.
    decoder : :class:`blocks.bricks.Brick`
        Decoder network.
    discriminator : :class:`blocks.bricks.Brick`
        Discriminator network taking :math:`x` and :math:`z` as input.
    n_cond: `int`
        Dimensionality of conditional data
    n_emb: `int`
        Dimensionality of embedding

    """
    def __init__(self, encoder, decoder, discriminator, n_cond, n_emb, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.n_cond = n_cond  # Features in conditional data
        self.n_emb = n_emb  # Features in embeddings
        self.embedder = Embedder(n_cond, n_emb, output_type='conv')

        super(ConditionalALI, self).__init__(**kwargs)
        self.children.extend([self.encoder, self.decoder, self.discriminator,
                              self.embedder])

    @property
    def discriminator_parameters(self):
        return list(
            Selector([self.discriminator]).get_parameters().values())

    @property
    def generator_parameters(self):
        return list(
            Selector([self.encoder, self.decoder]).get_parameters().values())
    @property
    def embedding_parameters(self):
        return list(
            Selector([self.embedder]).get_parameters().values())

    @application(inputs=['x', 'z_hat', 'x_tilde', 'z', 'y'],
                 outputs=['data_preds', 'sample_preds'])
    def get_predictions(self, x, z_hat, x_tilde, z, y, application_call):
        # NOTE: the unbroadcasts act as a workaround for a weird broadcasting
        # bug when applying dropout
        input_x = tensor.unbroadcast(
            tensor.concatenate([x, x_tilde], axis=0), *range(x.ndim))
        input_z = tensor.unbroadcast(
            tensor.concatenate([z_hat, z], axis=0), *range(x.ndim))
        input_y = tensor.unbroadcast(tensor.concatenate([y, y], axis=0), *range(x.ndim))
        data_sample_preds = self.discriminator.apply(input_x, input_z, input_y)
        data_preds = data_sample_preds[:x.shape[0]]
        sample_preds = data_sample_preds[x.shape[0]:]

        application_call.add_auxiliary_variable(
            tensor.nnet.sigmoid(data_preds).mean(), name='data_accuracy')
        application_call.add_auxiliary_variable(
            (1 - tensor.nnet.sigmoid(sample_preds)).mean(),
            name='sample_accuracy')

        return data_preds, sample_preds

    @application(inputs=['x', 'z', 'y'],
                 outputs=['discriminator_loss', 'generator_loss'])
    def compute_losses(self, x, z, y, application_call):
        embeddings = self.embedder.apply(y)
        z_hat = self.encoder.apply(x, embeddings)
        x_tilde = self.decoder.apply(z, embeddings)

        data_preds, sample_preds = self.get_predictions(x, z_hat, x_tilde, z,
                                                        embeddings)

        # To be modularized
        discriminator_loss = (tensor.nnet.softplus(-data_preds) +
                              tensor.nnet.softplus(sample_preds)).mean()
        generator_loss = (tensor.nnet.softplus(data_preds) +
                          tensor.nnet.softplus(-sample_preds)).mean()

        return discriminator_loss, generator_loss

    @application(inputs=['z', 'y'], outputs=['samples'])
    def sample(self, z, y):
        return self.decoder.apply(z, self.embedder.apply(y))

    @application(inputs=['x', 'y'], outputs=['reconstructions'])
    def reconstruct(self, x, y):
        embeddings = self.embedder.apply(y)
        return self.decoder.apply(self.encoder.apply(x, embeddings),
                                  embeddings)


if __name__ == '__main__':
    import numpy as np
    import numpy.random as npr

    WEIGHTS_INIT = IsotropicGaussian(0.01)
    BIASES_INIT = Constant(0.)
    LEAK = 0.1
    NLAT = 64

    IMAGE_SIZE = (32, 32)
    NUM_CHANNELS = 3
    NUM_PIECES = 2

    NCLASSES = 10
    NEMB = 100
    # Testing embedder
    embedder = Embedder(NCLASSES, NEMB, output_type='conv',
                        weights_init=WEIGHTS_INIT, biases_init=BIASES_INIT)
    embedder.initialize()

    x = tensor.tensor4('x')
    y = tensor.matrix('y')

    embedder_test = function([y], embedder.apply(y))

    test_labels = np.zeros(shape=(5, 10))
    idx = npr.randint(0, 9, size=5)
    for n, id in enumerate(idx):
        test_labels[n, id] = 1
    embeddings = embedder_test(test_labels)
    print(embeddings)
    print(embeddings.shape)

    # Generate synthetic 4D tensor
    features = npr.random(size=(5, 3, 32, 32))

    # Testing Encoder
    layers = [
        # 32 X 32 X 3
        conv_brick(5, 1, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        # 28 X 28 X 32
        conv_brick(4, 2, 64), bn_brick(), LeakyRectifier(leak=LEAK),
        # 13 X 13 X 64
        conv_brick(4, 1, 128), bn_brick(), LeakyRectifier(leak=LEAK),
        # 10 X 10 X 128
        conv_brick(4, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
        # 4 X 4 X 256
        conv_brick(4, 1, 512), bn_brick(), LeakyRectifier(leak=LEAK),
        # 1 X 1 X 512
        conv_brick(1, 1, 512), bn_brick(), LeakyRectifier(leak=LEAK),
        # 1 X 1 X 512
        conv_brick(1, 1, 2 * NLAT)
        # 1 X 1 X 2 * NLAT
    ]

    encoder_mapping = EncoderMapping(layers=layers,
                                     num_channels=NUM_CHANNELS,
                                     image_size=IMAGE_SIZE, weights_init=WEIGHTS_INIT,
                                     biases_init=BIASES_INIT)
    encoder_mapping.initialize()

    embeddings = embedder.apply(y)
    encoder_mapping_fun = function([x, y], encoder_mapping.apply(x, embeddings))
    out = encoder_mapping_fun(features, test_labels)
    print(out.shape)

    ## Testing Gaussian encoder blocks
    embeddings = embedder.apply(y)
    encoder = GaussianConditional(mapping=encoder_mapping)
    encoder.initialize()
    encoder_fun = function([x, y], encoder.apply(x, embeddings))
    z_hat = encoder_fun(features, test_labels)
    # print(out)
    print(z_hat)

    # Decoder
    z = tensor.tensor4('z')
    layers = [
        conv_transpose_brick(4, 1, 256), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(4, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(4, 1, 64), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(4, 2, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(5, 1, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(1, 1, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, NUM_CHANNELS), Logistic()]

    decoder = Decoder(layers=layers, num_channels=(NLAT + NEMB), image_size=(1, 1),
                      weights_init=WEIGHTS_INIT, biases_init=BIASES_INIT)
    decoder.initialize()
    decoder_fun = function([z, y], decoder.apply(z, embeddings))
    out = decoder_fun(z_hat, test_labels)

    # Discriminator

    layers = [
        conv_brick(5, 1, 32), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(4, 2, 64), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(4, 1, 128), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(4, 2, 256), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(4, 1, 512), ConvMaxout(num_pieces=NUM_PIECES)]
    x_discriminator = ConvolutionalSequence(
        layers=layers, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE,
        name='x_discriminator')
    x_discriminator.push_allocation_config()

    layers = [
        conv_brick(1, 1, 512), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(1, 1, 512), ConvMaxout(num_pieces=NUM_PIECES)]
    z_discriminator = ConvolutionalSequence(
        layers=layers, num_channels=NLAT, image_size=(1, 1), use_bias=False,
        name='z_discriminator')
    z_discriminator.push_allocation_config()

    layers = [
        conv_brick(1, 1, 1024), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(1, 1, 1024), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(1, 1, 1)]
    joint_discriminator = ConvolutionalSequence(
        layers=layers,
        num_channels=(x_discriminator.get_dim('output')[0] +
                      z_discriminator.get_dim('output')[0] +
                      NEMB),
        image_size=(1, 1),
        name='joint_discriminator')

    discriminator = XZYJointDiscriminator(
        x_discriminator, z_discriminator, joint_discriminator,
        name='discriminator')

    discriminator = XZYJointDiscriminator(x_discriminator, z_discriminator, joint_discriminator,
                                          name='discriminator', weights_init=WEIGHTS_INIT,
                                          biases_init=BIASES_INIT)
    discriminator.initialize()
    discriminator_fun = function([x, z, y], discriminator.apply(x, z, embeddings))
    out = discriminator_fun(features, z_hat, test_labels)
    print(out.shape)


    # Initializing ALI
    ali = ConditionalALI(encoder=encoder, decoder=decoder, discriminator=discriminator,
                         n_cond=NCLASSES,
                         n_emb=NEMB,
                         weights_init=WEIGHTS_INIT,
                         biases_init=BIASES_INIT)
    ali.initialize()
    # Computing Loss
    loss = ali.compute_losses(x, z, y)
    loss_fun = function([x, z, y], loss)
    out = loss_fun(features, z_hat, test_labels)


