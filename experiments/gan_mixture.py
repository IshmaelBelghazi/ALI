import itertools

import numpy
from theano import tensor
from blocks.algorithms import Adam
from blocks.bricks import MLP, Rectifier, Identity, LinearMaxout, Linear
from blocks.bricks.bn import BatchNormalization
from blocks.bricks.sequences import Sequence
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph, apply_dropout
from blocks.graph.bn import (batch_normalization,
                             get_batch_normalization_updates)
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from blocks.main_loop import MainLoop
from blocks.roles import INPUT

from ali.algorithms import ali_algorithm
from ali.streams import create_gaussian_mixture_data_streams
from ali.bricks import GAN
from ali.utils import as_array

import logging
import argparse

INPUT_DIM = 2
NLAT = 2
GEN_HIDDEN = 400
DISC_HIDDEN = 200
GEN_ACTIVATION = Rectifier
MAXOUT_PIECES = 5
GAUSSIAN_INIT = IsotropicGaussian(std=0.02)
ZERO_INIT = Constant(0.0)

NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BETA1 = 0.8
BATCH_SIZE = 100
MONITORING_BATCH_SIZE = 500
MEANS = [numpy.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                           range(-4, 5, 2))]
VARIANCES = [0.05 ** 2 * numpy.eye(len(mean)) for mean in MEANS]
PRIORS = None


def create_model_brick():
    decoder = MLP(
        dims=[NLAT, GEN_HIDDEN, GEN_HIDDEN, GEN_HIDDEN, GEN_HIDDEN, INPUT_DIM],
        activations=[Sequence([BatchNormalization(GEN_HIDDEN).apply,
                               GEN_ACTIVATION().apply],
                              name='decoder_h1'),
                     Sequence([BatchNormalization(GEN_HIDDEN).apply,
                               GEN_ACTIVATION().apply],
                              name='decoder_h2'),
                     Sequence([BatchNormalization(GEN_HIDDEN).apply,
                               GEN_ACTIVATION().apply],
                              name='decoder_h3'),
                     Sequence([BatchNormalization(GEN_HIDDEN).apply,
                               GEN_ACTIVATION().apply],
                              name='decoder_h4'),
                     Identity(name='decoder_out')],
        use_bias=False,
        name='decoder')

    discriminator = Sequence(
        application_methods=[
            LinearMaxout(
                input_dim=INPUT_DIM,
                output_dim=DISC_HIDDEN,
                num_pieces=MAXOUT_PIECES,
                weights_init=GAUSSIAN_INIT,
                biases_init=ZERO_INIT,
                name='discriminator_h1').apply,
            LinearMaxout(
                input_dim=DISC_HIDDEN,
                output_dim=DISC_HIDDEN,
                num_pieces=MAXOUT_PIECES,
                weights_init=GAUSSIAN_INIT,
                biases_init=ZERO_INIT,
                name='discriminator_h2').apply,
            LinearMaxout(
                input_dim=DISC_HIDDEN,
                output_dim=DISC_HIDDEN,
                num_pieces=MAXOUT_PIECES,
                weights_init=GAUSSIAN_INIT,
                biases_init=ZERO_INIT,
                name='discriminator_h3').apply,
            Linear(
                input_dim=DISC_HIDDEN,
                output_dim=1,
                weights_init=GAUSSIAN_INIT,
                biases_init=ZERO_INIT,
                name='discriminator_out').apply],
        name='discriminator')

    gan = GAN(decoder=decoder, discriminator=discriminator,
              weights_init=GAUSSIAN_INIT, biases_init=ZERO_INIT, name='gan')
    gan.push_allocation_config()
    decoder.linear_transformations[-1].use_bias = True
    gan.initialize()

    return gan


def create_models():
    gan = create_model_brick()
    x = tensor.matrix('features')
    z = gan.theano_rng.normal(size=(x.shape[0], NLAT))

    def _create_model(with_dropout):
        cg = ComputationGraph(gan.compute_losses(x, z))
        if with_dropout:
            inputs = VariableFilter(
                bricks=gan.discriminator.children[1:],
                roles=[INPUT])(cg.variables)
            cg = apply_dropout(cg, inputs, 0.5)
            inputs = VariableFilter(
                bricks=[gan.discriminator],
                roles=[INPUT])(cg.variables)
            cg = apply_dropout(cg, inputs, 0.2)
        return Model(cg.outputs)

    model = _create_model(with_dropout=False)
    with batch_normalization(gan):
        bn_model = _create_model(with_dropout=False)

    pop_updates = list(
        set(get_batch_normalization_updates(bn_model, allow_duplicates=True)))
    bn_updates = [(p, m * 0.05 + p * 0.95) for p, m in pop_updates]

    return model, bn_model, bn_updates


def create_main_loop(save_path):
    model, bn_model, bn_updates = create_models()
    gan, = bn_model.top_bricks
    discriminator_loss, generator_loss = bn_model.outputs
    step_rule = Adam(learning_rate=LEARNING_RATE, beta1=BETA1)
    algorithm = ali_algorithm(discriminator_loss, gan.discriminator_parameters,
                              step_rule, generator_loss,
                              gan.generator_parameters, step_rule)
    algorithm.add_updates(bn_updates)
    streams = create_gaussian_mixture_data_streams(
        batch_size=BATCH_SIZE, monitoring_batch_size=MONITORING_BATCH_SIZE,
        means=MEANS, variances=VARIANCES, priors=PRIORS)
    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams
    bn_monitored_variables = (
        [v for v in bn_model.auxiliary_variables if 'norm' not in v.name] +
        bn_model.outputs)
    monitored_variables = (
        [v for v in model.auxiliary_variables if 'norm' not in v.name] +
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
    parser = argparse.ArgumentParser(description='Train GAN on MOG')
    parser.add_argument("--save-path", type=str,
                        default='gan_mixture_prime.tar',
                        help="main loop save path")
    args = parser.parse_args()
    main_loop = create_main_loop(args.save_path)
    main_loop.run()
