import argparse
import logging
import random
import string
from collections import OrderedDict

import numpy
import numpy as np
import pylab as plb

import theano
from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import Rectifier, Logistic, Identity, Random
from blocks.bricks.bn import SpatialBatchNormalization
from blocks.bricks.conv import (Convolutional, ConvolutionalTranspose,
                                ConvolutionalSequence)
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
from blocks.select import Selector
from fuel.datasets import SVHN
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from theano import tensor

from research.toolbox.datasets import STREAMS
from research.toolbox.tools import as_array, load_mainloop
from ali import GAN
from ali_models import (create_generator, create_maxout_discriminator, )

input_dim = 2
nlat = 2
gaussian_init = IsotropicGaussian(std=0.02)
zero_init = Constant(0.0)
gen_activation = Rectifier
gen_hidden = 1000
disc_hidden = 400

batch_size = 100
monitoring_batch_size = 500
num_epochs = 5
increments = 1000 // 5

def plot_spiral(main_loop, train_data, Z, x_tilde):
    """
    Plots a scatter plot of the spiral
    """
    square_width = 4

    current_epoch = main_loop.status['epochs_done']
    plb.suptitle('Epoch: {}'.format(current_epoch))
    plb.title('Samples')
    # Collecting train dataset
    # PLotting Visible space
    visible_ax = plb.subplot(211)
    visible_ax.clear()
    visible_ax.set_title('X space')
    visible_ax.set_xlim(xmin=-square_width, xmax=square_width)
    visible_ax.set_ylim(ymin=-square_width, ymax=square_width)
    # Train data
    plb.scatter(x=train_data[:, 0], y=train_data[:, 1],
                marker='o', c='black', alpha=0.3)
    # Plotting Samples
    plb.scatter(x=x_tilde[:, 0], y=x_tilde[:, 1],
                marker='x', c='blue', alpha=0.3)

    # Hiddem
    hidden_ax = plb.subplot(212)
    hidden_ax.clear()
    hidden_ax.set_title('Z space')
    hidden_ax.set_xlim(xmin=-square_width, xmax=square_width)
    hidden_ax.set_ylim(ymin=-square_width, ymax=square_width)
    # Train data
    # Plotting Samples
    plb.scatter(x=Z[:, 0], y=Z[:, 1],
                marker='x', c='blue', alpha=0.3, label='Z')
    plb.title('Prior')
    plb.draw()
    plb.pause(0.05)
    filename = 'epoch_' + str(current_epoch) + '_' + 'gan_spiral_plot.png'
    plb.savefig(filename)
    # plb.show()

def name_generator():
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(8))


def create_sampling_computation_graph(main_loop, num_samples):
    gan, = main_loop.model.top_bricks
    input_shape = gan.decoder.get_dim('input_')
    z = gan.theano_rng.normal(size=(num_samples,) + input_shape)
    x = gan.sample(z)
    return ComputationGraph([x])


def create_data_streams(batch_size, monitoring_batch_size, rng=None):
    main_loop_stream = STREAMS['SPIRAL']['streams'](sources=('features', ),
                                                    batch_size=batch_size)['train']
    train_monitor_stream = STREAMS['SPIRAL']['streams'](sources=('features', ),
                                                        batch_size=monitoring_batch_size)['train']
    valid_monitor_stream = STREAMS['SPIRAL']['streams'](sources=('features', ),
                                                        batch_size=monitoring_batch_size)['test']
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def get_log_odds():
    main_loop_stream = create_data_streams(500, 500)[0]
    raw_train_data, = next(main_loop_stream.get_epoch_iterator())
    marginals = numpy.clip(raw_train_data.mean(axis=0), 1e-7, 1 - 1e-7)
    return numpy.log(marginals / (1 - marginals)).astype(theano.config.floatX)


def create_model_bricks():
    _, decoder = create_generator(input_dim=input_dim,
                                  hidden_activation=gen_activation,
                                  n_hidden=gen_hidden,
                                  nlat=nlat,
                                  weights_init=gaussian_init,
                                  biases_init=zero_init,
                                  output_act=Identity)

    discriminator = create_maxout_discriminator(input_dim=input_dim,
                                                nlat=0, # nlat is input dim
                                                # with GANs
                                                n_hidden=disc_hidden,
                                                num_pieces=2,
                                                weights_init=gaussian_init,
                                                biases_init=zero_init)

    # GAN
    gan = GAN(decoder, discriminator, weights_init=gaussian_init,
              biases_init=zero_init, name='gan')
    gan.push_allocation_config()
    gan.initialize()
    return gan


def create_models():
    gan = create_model_bricks()
    x = tensor.matrix('features')
    z = gan.theano_rng.normal(size=(x.shape[0], nlat))

    def _create_model(with_dropout):
        cg = ComputationGraph(gan.compute_losses(x, z))
        if with_dropout:
            child_idx = [n for n, child in enumerate(gan.discriminator.children) if 'linear' in child.name]
            d_upper_linear = [gan.discriminator.children[n] for n in child_idx[1:]]
            upper_disc_inputs = VariableFilter(bricks=d_upper_linear,
                                               roles=[INPUT])(cg.variables)
            interm_cg = apply_dropout(cg, upper_disc_inputs, 0.5)
            discriminator_inputs = VariableFilter(bricks=[gan.discriminator],
                                                  roles=[INPUT])(interm_cg)
            cg = apply_dropout(interm_cg, discriminator_inputs, 0.2)
        return Model(cg.outputs)

    model = _create_model(with_dropout=False)
    with batch_normalization(gan):
        bn_model = _create_model(with_dropout=True)

    pop_updates = list(
        set(get_batch_normalization_updates(bn_model, allow_duplicates=True)))
    bn_updates = [(p, m * 0.05 + p * 0.95) for p, m in pop_updates]

    return model, bn_model, bn_updates


def create_main_loop(save_path):

    model, bn_model, bn_updates = create_models()
    gan, = bn_model.top_bricks

    discriminator_loss, generator_loss = bn_model.outputs
    dummy_total_loss = generator_loss + discriminator_loss

    grads = OrderedDict()
    grads.update(zip(gan.discriminator_parameters,
                     theano.grad(discriminator_loss,
                                 gan.discriminator_parameters)))
    grads.update(zip(gan.generator_parameters,
                     theano.grad(generator_loss,
                                 gan.generator_parameters)))

    step_rule = Adam(learning_rate=0.001)

    algorithm = GradientDescent(
        cost=dummy_total_loss, gradients=grads, parameters=bn_model.parameters,
        step_rule=step_rule)
    algorithm.add_updates(bn_updates)
    streams = create_data_streams(batch_size, monitoring_batch_size)
    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams
    bn_monitored_variables = (
        [v for v in bn_model.auxiliary_variables if v.name.startswith('gan')] +
        bn_model.outputs)
    monitored_variables = (
        [v for v in model.auxiliary_variables if v.name.startswith('gan')] +
        model.outputs)
    extensions = [
        Timing(),
        FinishAfter(every_n_epochs=num_epochs),
        DataStreamMonitoring(
            bn_monitored_variables, train_monitor_stream, prefix="train",
            updates=bn_updates),
        DataStreamMonitoring(
            monitored_variables, valid_monitor_stream, prefix="valid"),
        Checkpoint(save_path, after_epoch=True, after_training=True,
                   use_cpickle=True),
        ProgressBar(),
        Printing()]
    main_loop = MainLoop(model=bn_model, data_stream=main_loop_stream,
                         algorithm=algorithm, extensions=extensions)
    return main_loop


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_path = "/Tmp/belghamo/gan_spiral.tar"
    # main_loop = create_main_loop(save_path)
    main_loop = load_mainloop(save_path)

    # Compiling sampling function
    gan, = Selector(main_loop.model.top_bricks).select('/gan').bricks

    Z = tensor.matrix('samples')
    gan_sampler = theano.function([Z], gan.sample_x_tilde(Z))
    # Getting data

    train_stream = STREAMS['SPIRAL']['streams'](sources=('features', ),
                                                batch_size=1000)['train']
    valid_stream = STREAMS['SPIRAL']['streams'](sources=('features', ),
                                                batch_size=1000)['test']
    train_data, = next(train_stream.get_epoch_iterator())
    valid_data, = next(train_stream.get_epoch_iterator())

    # Getting Z
    Z = as_array(np.random.normal(size=(1000, 2)))
    X_tilde = gan_sampler(Z)
    plot_spiral(main_loop, train_data, Z, X_tilde)
