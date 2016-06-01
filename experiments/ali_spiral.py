import argparse
import logging

from blocks.algorithms import Adam
from blocks.bricks import MLP, Rectifier, Identity, LinearMaxout, Linear
from blocks.bricks.bn import BatchNormalization
from blocks.bricks.sequences import Sequence
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.graph.bn import (batch_normalization,
                             get_batch_normalization_updates)
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import INPUT
from theano import tensor

from ali.algorithms import ali_algorithm
from ali.bricks import FullyConnectedALI
from ali.streams import create_spiral_data_streams

INPUT_DIM = 2
NLAT = 2
GAUSSIAN_INIT = IsotropicGaussian(std=0.02)
ZERO_INIT = Constant(0.0)
MAXOUT_PIECES = 2
GEN_HIDDEN = 1000
GEN_ACTIVATION = Rectifier
DISC_HIDDEN = 400
DISC_ACTIVATION = Rectifier
LEARNING_RATE = 0.001
NUM_EPOCHS = 200
BATCH_SIZE = 100
MONITORING_BATCH_SIZE = 500


def create_model_brick():
    encoder = MLP(
        dims=[INPUT_DIM, GEN_HIDDEN, GEN_HIDDEN, 2 * NLAT],
        activations=[Sequence([BatchNormalization(GEN_HIDDEN).apply,
                               GEN_ACTIVATION().apply],
                              name='encoder_h1'),
                     Sequence([BatchNormalization(GEN_HIDDEN).apply,
                               GEN_ACTIVATION().apply],
                              name='encoder_h2'),
                     Identity(name='encoder_out')],
        use_bias=False,
        name='encoder')

    decoder = MLP(
        dims=[NLAT, GEN_HIDDEN, GEN_HIDDEN, INPUT_DIM],
        activations=[Sequence([BatchNormalization(GEN_HIDDEN).apply,
                               GEN_ACTIVATION().apply],
                              name='decoder_h1'),
                     Sequence([BatchNormalization(GEN_HIDDEN).apply,
                               GEN_ACTIVATION().apply],
                              name='decoder_h2'),
                     Identity(name='decoder_out')],
        use_bias=False,
        name='decoder')

    discriminator = Sequence(
        application_methods=[
            LinearMaxout(
                input_dim=INPUT_DIM + NLAT,
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
            Linear(
                input_dim=DISC_HIDDEN,
                output_dim=1,
                weights_init=GAUSSIAN_INIT,
                biases_init=ZERO_INIT,
                name='discriminator_out').apply],
        name='discriminator')

    ali = FullyConnectedALI(
        encoder=encoder, decoder=decoder, discriminator=discriminator,
        weights_init=GAUSSIAN_INIT, biases_init=ZERO_INIT, name='ali')
    ali.push_allocation_config()
    encoder.linear_transformations[-1].use_bias = True
    decoder.linear_transformations[-1].use_bias = True
    ali.initialize()

    return ali


def create_models():
    ali = create_model_brick()
    x = tensor.matrix('features')
    z = ali.theano_rng.normal(size=(x.shape[0], ali.decoder.input_dim))

    def _create_model(with_dropout):
        cg = ComputationGraph(ali.compute_losses(x, z))
        if with_dropout:
            inputs = VariableFilter(
                bricks=ali.discriminator.children[1:],
                roles=[INPUT])(cg.variables)
            cg = apply_dropout(cg, inputs, 0.5)
            inputs = VariableFilter(
                bricks=[ali.discriminator], roles=[INPUT])(cg)
            cg = apply_dropout(cg, inputs, 0.2)

        return Model(cg.outputs)

    model = _create_model(with_dropout=False)
    with batch_normalization(ali):
        bn_model = _create_model(with_dropout=True)

    pop_updates = list(
        set(get_batch_normalization_updates(bn_model, allow_duplicates=True)))
    bn_updates = [(p, m * 0.05 + p * 0.95) for p, m in pop_updates]

    return model, bn_model, bn_updates


def create_main_loop(save_path):
    model, bn_model, bn_updates = create_models()
    ali, = bn_model.top_bricks
    discriminator_loss, generator_loss = bn_model.outputs
    step_rule = Adam(learning_rate=LEARNING_RATE)
    algorithm = ali_algorithm(discriminator_loss, ali.discriminator_parameters,
                              step_rule, generator_loss,
                              ali.generator_parameters, step_rule)
    algorithm.add_updates(bn_updates)
    streams = create_spiral_data_streams(BATCH_SIZE, MONITORING_BATCH_SIZE)

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
    create_main_loop(args.save_path).run()
