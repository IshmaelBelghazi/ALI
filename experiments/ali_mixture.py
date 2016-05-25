
from theano import tensor
from blocks.algorithms import Adam
from blocks.bricks import (MLP, Rectifier, Logistic, Identity, LinearMaxout,
                           Linear, Tanh, LeakyRectifier, )
from blocks.bricks.bn import (BatchNormalization, )
from blocks.bricks.sequences import Sequence
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph, apply_dropout
from blocks.graph.bn import (batch_normalization,
                             get_batch_normalization_updates, )
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from blocks.main_loop import MainLoop
from blocks.roles import INPUT

from ali.algorithms import ali_algorithm
from ali.streams import create_gaussian_mixture_data_streams
from ali.bricks import FullyConnectedALI
from ali.utils import as_array

import logging
import argparse

# Dimensions
INPUT_DIM = 2
NLAT = 2
# Initialization
GAUSSIAN_INIT = IsotropicGaussian(std=0.02)
ZERO_INIT = Constant(0.0)
# Model
BATCH_SIZE = 100
GEN_HIDDEN = 1000
GEN_ACTIVATION = Rectifier #Rectifier #partial(LeakyRectifier, leak=0.2)
DISC_HIDDEN = 400
# DISC_ACTIVATION = Rectifier
MAXOUT_PIECES = 2
WITH_DROPOUT = True

# Optimization
LEARNING_RATE = 0.001
BETA1 = 0.5
BATCH_SIZE = 100
NUM_EPOCHS = 1000
# Monitoring
MONITORING_BATCH_SIZE = 500
SEED = None
# Dataset
MEANS = map(lambda x:  10.0 * as_array(x), [[0, 0],
                                            [1, 1],
                                            [-1, -1],
                                            [1, -1],
                                            [-1, 1]])
VARIANCES = None  # Identity covariances.
PRIORS = None  # Equally Likely


def create_generator(input_dim, hidden_activation,
                     n_hidden, nlat,
                     weights_init, biases_init,
                     linear_init=None,
                     output_act=Logistic):

    # Encoder
    encoder = MLP(
        dims=[input_dim, n_hidden, n_hidden, 2 * nlat],
        activations=[Sequence([BatchNormalization(n_hidden).apply,
                               hidden_activation().apply],
                              name='encoder_h1'),
                     Sequence([BatchNormalization(n_hidden).apply,
                               hidden_activation().apply],
                               name='encoder_h2'),
                     Identity(name='encoder_out')],
        use_bias=False,
        weights_init=weights_init,
        biases_init=biases_init,
        name='encoder')
    encoder.push_allocation_config()
    encoder.linear_transformations[-1].use_bias = True
    if linear_init is not None:
        encoder.linear_transformations[-1].weights_init = linear_init
    encoder.initialize()

    # Decoder
    decoder = MLP(
        dims=[nlat, n_hidden, n_hidden, input_dim],
        activations=[Sequence([BatchNormalization(n_hidden).apply,
                               hidden_activation().apply],
                              name='decoder_h1'),
                     Sequence([BatchNormalization(n_hidden).apply,
                               hidden_activation().apply],
                              name='decoder_h2'),
                     output_act(name='decoder_out')],
        use_bias=False,
        weights_init=weights_init,
        biases_init=biases_init,
        name='decoder')

    decoder.push_allocation_config()
    decoder.linear_transformations[-1].use_bias = True
    decoder.initialize()

    return encoder, decoder


def create_discriminator(input_dim, nlat,
                         hidden_activation,
                         n_hidden,
                         weights_init,
                         biases_init,
                         output_init=None):

    # Specifying discriminator
    # Dimensions
    dims=[input_dim + nlat, n_hidden, n_hidden, 1]
    # Activations
    activations = [Sequence([hidden_activation().apply],
                            name='discriminator_h1'),
                   Sequence([hidden_activation().apply],
                            name='discriminator_h2'),
                   Linear(name='discriminator_out')]

    discriminator = MLP(dims=dims,
                        activations=activations,
                        use_bias=True,
                        weights_init=weights_init,
                        biases_init=biases_init,
                        name='discriminator')

    discriminator.push_allocation_config()
    if output_init is not None:
        discriminator.linear_transformations[-1].weights_init = output_init
    discriminator.initialize()

    return discriminator


def create_maxout_discriminator(input_dim, nlat,
                                n_hidden,
                                num_pieces,
                                weights_init,
                                biases_init,
                                output_init=None):

    discriminator = Sequence(
        application_methods=[
            LinearMaxout(
                input_dim=input_dim + nlat,
                output_dim=n_hidden,
                num_pieces=num_pieces,
                weights_init=weights_init,
                biases_init=biases_init,
                name='discriminator_h1').apply,
            LinearMaxout(
                input_dim=n_hidden,
                output_dim=n_hidden,
                num_pieces=num_pieces,
                weights_init=weights_init,
                biases_init=biases_init,
                name='discriminator_h2').apply,
            Linear(
                input_dim=n_hidden,
                output_dim=1,
                weights_init=weights_init if output_init is None else output_init,
                biases_init=biases_init,
                name='discriminator_out').apply],
        name='discriminator')

    discriminator.push_allocation_config()
    discriminator.initialize()

    return discriminator


def create_model_brick():
    encoder, decoder = create_generator(input_dim=INPUT_DIM,
                                        hidden_activation=GEN_ACTIVATION,
                                        n_hidden=GEN_HIDDEN, nlat=NLAT,
                                        weights_init=GAUSSIAN_INIT,
                                        biases_init=ZERO_INIT,
                                        output_act=Identity)

    discriminator = create_maxout_discriminator(input_dim=INPUT_DIM,
                                                nlat=NLAT,
                                                num_pieces=MAXOUT_PIECES,
                                                n_hidden=DISC_HIDDEN,
                                                weights_init=GAUSSIAN_INIT,
                                                biases_init=ZERO_INIT)

    ali = FullyConnectedALI(encoder=encoder, decoder=decoder,
                            discriminator=discriminator)
    ali.initialize()
    return ali


def create_models():
    ali = create_model_brick()
    x = tensor.matrix('features')
    z = ali.theano_rng.normal(
        size=(x.shape[0], ali.decoder.input_dim))

    def _create_model(with_dropout):
        cg = ComputationGraph(ali.compute_losses(x, z))
        if with_dropout:
            child_idx = [n for n, child in enumerate(
                ali.discriminator.children) if 'linear' in child.name]
            d_upper_linear = [ali.discriminator.children[n] for n in child_idx[1:]]
            upper_disc_inputs = VariableFilter(bricks=d_upper_linear,
                                               roles=[INPUT])(cg.variables)
            interm_cg = apply_dropout(cg, upper_disc_inputs, 0.5)
            discriminator_inputs = VariableFilter(bricks=[ali.discriminator],
                                                  roles=[INPUT])(interm_cg)
            cg = apply_dropout(interm_cg, discriminator_inputs, 0.2)

        return Model(cg.outputs)

    model = _create_model(with_dropout=False)
    with batch_normalization(ali):
        bn_model = _create_model(with_dropout=WITH_DROPOUT)

    pop_updates = list(
        set(get_batch_normalization_updates(bn_model, allow_duplicates=True)))
    bn_updates = [(p, m * 0.05 + p * 0.95) for p, m in pop_updates]

    return model, bn_model, bn_updates


def create_main_loop(save_path):
    model, bn_model, bn_updates = create_models()
    ali, = bn_model.top_bricks
    discriminator_loss, generator_loss = bn_model.outputs
    step_rule = Adam(learning_rate=LEARNING_RATE, beta1=BETA1)
    algorithm = ali_algorithm(discriminator_loss, ali.discriminator_parameters,
                              step_rule, generator_loss,
                              ali.generator_parameters, step_rule)
    algorithm.add_updates(bn_updates)
    streams = create_gaussian_mixture_data_streams(batch_size=BATCH_SIZE,
                                                   monitoring_batch_size=MONITORING_BATCH_SIZE,
                                                   means=MEANS,
                                                   variances=VARIANCES,
                                                   priors=PRIORS)

    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams
    bn_monitored_variables = (
        [v for v in bn_model.auxiliary_variables if v.name.startswith('ali')] +
        bn_model.outputs)
    monitored_variables = (
        [v for v in model.auxiliary_variables if v.name.startswith('ali')] +
        model.outputs)
    extensions = [
        Timing(),
        FinishAfter(after_n_epochs=NUM_EPOCHS),
        DataStreamMonitoring(
            bn_monitored_variables, train_monitor_stream, prefix="train",
            updates=bn_updates),
        DataStreamMonitoring(
            monitored_variables, valid_monitor_stream, prefix="valid"),
        Checkpoint(save_path, after_epoch=True, after_training=True,
                   use_cpickle=True),
        ProgressBar(),
        Printing(),
    ]
    main_loop = MainLoop(model=bn_model, data_stream=main_loop_stream,
                         algorithm=algorithm, extensions=extensions)
    return main_loop

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Train ALI on Spiral')
    parser.add_argument("--save-path", type=str, default='ali_spiral.tar',
                        help="main loop save path")
    args = parser.parse_args()
    main_loop = create_main_loop(args.save_path)
    main_loop.run()
