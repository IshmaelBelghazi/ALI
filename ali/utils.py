"""Utility functions."""
import random
import string

import numpy
import theano
from blocks.bricks.bn import SpatialBatchNormalization
from blocks.bricks.conv import Convolutional, ConvolutionalTranspose


def name_generator():
    """Returns a random 8-character name."""
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(8))


def get_log_odds(raw_marginals):
    """Computes marginal log-odds."""
    marginals = numpy.clip(raw_marginals.mean(axis=0), 1e-7, 1 - 1e-7)
    return numpy.log(marginals / (1 - marginals)).astype(theano.config.floatX)


def conv_brick(filter_size, step, num_filters, border_mode='valid'):
    """Instantiates a ConvolutionalBrick."""
    return Convolutional(filter_size=(filter_size, filter_size),
                         step=(step, step),
                         border_mode=border_mode,
                         num_filters=num_filters,
                         name=name_generator())


def conv_transpose_brick(filter_size, step, num_filters, border_mode='valid'):
    """Instantiates a ConvolutionalTranspose brick."""
    return ConvolutionalTranspose(filter_size=(filter_size, filter_size),
                                  step=(step, step),
                                  border_mode=border_mode,
                                  num_filters=num_filters,
                                  name=name_generator())


def bn_brick():
    """Instantiates a SpatialBatchNormalization brick."""
    return SpatialBatchNormalization(name=name_generator())


def as_array(obj, dtype=theano.config.floatX):
    """Converts to ndarray of specified dtype"""
    return numpy.asarray(obj, dtype=dtype)
