"""Functions for creating data streams."""
from fuel.datasets import CIFAR10, SVHN, CelebA
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

from .datasets import TinyILSVRC2012


def create_svhn_data_streams(batch_size, monitoring_batch_size, rng=None):
    train_set = SVHN(2, ('extra',), sources=('features',))
    valid_set = SVHN(2, ('train',), sources=('features',))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_cifar10_data_streams(batch_size, monitoring_batch_size, rng=None):
    train_set = CIFAR10(
        ('train',), sources=('features',), subset=slice(0, 45000))
    valid_set = CIFAR10(
        ('train',), sources=('features',), subset=slice(45000, 50000))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_celeba_data_streams(batch_size, monitoring_batch_size, rng=None):
    train_set = CelebA('64', ('train',), sources=('features',))
    valid_set = CelebA('64', ('valid',), sources=('features',))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_tiny_imagenet_data_streams(batch_size, monitoring_batch_size,
                                      rng=None):
    train_set = TinyILSVRC2012(('train',), sources=('features',))
    valid_set = TinyILSVRC2012(('valid',), sources=('features',))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            4096, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            4096, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream
