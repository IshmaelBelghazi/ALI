import argparse
import logging

from blocks.algorithms import Adam
from blocks.bricks import LeakyRectifier, Logistic
from blocks.bricks.conv import ConvolutionalSequence
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
from ali.bricks import (ALI, GaussianConditional, DeterministicConditional,
                        XZJointDiscriminator, ConvMaxout)
from ali.streams import create_cifar10_data_streams
from ali.utils import get_log_odds, conv_brick, conv_transpose_brick, bn_brick

BATCH_SIZE = 100
MONITORING_BATCH_SIZE = 500
NUM_EPOCHS = 6475
IMAGE_SIZE = (32, 32)
NUM_CHANNELS = 3
NLAT = 64
GAUSSIAN_INIT = IsotropicGaussian(std=0.01)
ZERO_INIT = Constant(0)
LEAK = 0.1
NUM_PIECES = 2
LEARNING_RATE = 1e-4
BETA1 = 0.5


def create_model_brick():
    layers = [
        conv_brick(5, 1, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(4, 2, 64), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(4, 1, 128), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(4, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(4, 1, 512), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, 512), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, 2 * NLAT)]
    encoder_mapping = ConvolutionalSequence(
        layers=layers, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE,
        use_bias=False, name='encoder_mapping')
    encoder = GaussianConditional(encoder_mapping, name='encoder')

    layers = [
        conv_transpose_brick(4, 1, 256), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(4, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(4, 1, 64), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(4, 2, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(5, 1, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(1, 1, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, NUM_CHANNELS), Logistic()]
    decoder_mapping = ConvolutionalSequence(
        layers=layers, num_channels=NLAT, image_size=(1, 1), use_bias=False,
        name='decoder_mapping')
    decoder = DeterministicConditional(decoder_mapping, name='decoder')

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
                      z_discriminator.get_dim('output')[0]),
        image_size=(1, 1),
        name='joint_discriminator')

    discriminator = XZJointDiscriminator(
        x_discriminator, z_discriminator, joint_discriminator,
        name='discriminator')

    ali = ALI(encoder, decoder, discriminator,
              weights_init=GAUSSIAN_INIT, biases_init=ZERO_INIT,
              name='ali')
    ali.push_allocation_config()
    encoder_mapping.layers[-1].use_bias = True
    encoder_mapping.layers[-1].tied_biases = False
    decoder_mapping.layers[-2].use_bias = True
    decoder_mapping.layers[-2].tied_biases = False
    ali.initialize()
    raw_marginals, = next(
        create_cifar10_data_streams(500, 500)[0].get_epoch_iterator())
    b_value = get_log_odds(raw_marginals)
    decoder_mapping.layers[-2].b.set_value(b_value)

    return ali


def create_models():
    ali = create_model_brick()
    x = tensor.tensor4('features')
    z = ali.theano_rng.normal(size=(x.shape[0], NLAT, 1, 1))

    def _create_model(with_dropout):
        cg = ComputationGraph(ali.compute_losses(x, z))
        if with_dropout:
            inputs = VariableFilter(
                bricks=([ali.discriminator.x_discriminator.layers[0],
                         ali.discriminator.z_discriminator.layers[0]]),
                roles=[INPUT])(cg.variables)
            cg = apply_dropout(cg, inputs, 0.2)
            inputs = VariableFilter(
                bricks=(ali.discriminator.x_discriminator.layers[2::3] +
                        ali.discriminator.z_discriminator.layers[2::2] +
                        ali.discriminator.joint_discriminator.layers[::2]),
                roles=[INPUT])(cg.variables)
            cg = apply_dropout(cg, inputs, 0.5)
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

    step_rule = Adam(learning_rate=LEARNING_RATE, beta1=BETA1)
    algorithm = ali_algorithm(discriminator_loss, ali.discriminator_parameters,
                              step_rule, generator_loss,
                              ali.generator_parameters, step_rule)
    algorithm.add_updates(bn_updates)
    streams = create_cifar10_data_streams(BATCH_SIZE, MONITORING_BATCH_SIZE)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train ALI on CIFAR10")
    parser.add_argument("--save-path", type=str, default='ali_cifar10.tar',
                        help="main loop save path")
    args = parser.parse_args()
    create_main_loop(args.save_path).run()
