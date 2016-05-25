import argparse

import numpy
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme, ShuffledExampleScheme
from fuel.streams import DataStream
from sklearn.svm import LinearSVC


def main(dataset_path, use_c, log_min, log_max, num_steps):
    train_set = H5PYDataset(
        dataset_path, which_sets=('train',), sources=('features', 'targets'),
        subset=slice(0, 63257), load_in_memory=True)
    train_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledExampleScheme(train_set.num_examples))

    def get_class_balanced_batch(iterator):
        train_features = [[] for _ in range(10)]
        train_targets = [[] for _ in range(10)]
        batch_size = 0
        while batch_size < 1000:
            f, t = next(iterator)
            t = t[0]
            if len(train_features[t]) < 100:
                train_features[t].append(f)
                train_targets[t].append(t)
                batch_size += 1
        train_features = numpy.vstack(sum(train_features, []))
        train_targets = numpy.vstack(sum(train_targets, []))
        return train_features, train_targets

    train_features, train_targets = get_class_balanced_batch(
        train_stream.get_epoch_iterator())

    valid_set = H5PYDataset(
        dataset_path, which_sets=('train',), sources=('features', 'targets'),
        subset=slice(63257, 73257), load_in_memory=True)
    valid_features, valid_targets = valid_set.data_sources

    test_set = H5PYDataset(
        dataset_path, which_sets=('test',), sources=('features', 'targets'),
        load_in_memory=True)
    test_features, test_targets = test_set.data_sources

    if use_c is None:
        best_error_rate = 1.0
        best_C = None
        for log_C in numpy.linspace(log_min, log_max, num_steps):
            C = numpy.exp(log_C)
            svm = LinearSVC(C=C)
            svm.fit(train_features, train_targets.ravel())
            error_rate = 1 - numpy.mean(
                [svm.score(valid_features[1000 * i: 1000 * (i + 1)],
                           valid_targets[1000 * i: 1000 * (i + 1)].ravel())
                 for i in range(10)])
            if error_rate < best_error_rate:
                best_error_rate = error_rate
                best_C = C
            print('C = {}, validation error rate = {} '.format(C, error_rate) +
                  '(best is {}, {})'.format(best_C, best_error_rate))
    else:
        best_C = use_c

    error_rates = []
    for _ in range(10):
        train_features, train_targets = get_class_balanced_batch(
            train_stream.get_epoch_iterator())
        svm = LinearSVC(C=best_C)
        svm.fit(train_features, train_targets.ravel())
        error_rates.append(1 - numpy.mean(
            [svm.score(valid_features[1000 * i: 1000 * (i + 1)],
                       valid_targets[1000 * i: 1000 * (i + 1)].ravel())
             for i in range(10)]))

    print('Validation error rate = {} +- {} '.format(numpy.mean(error_rates),
                                                     numpy.std(error_rates)))

    error_rates = []
    for _ in range(100):
        train_features, train_targets = get_class_balanced_batch(
            train_stream.get_epoch_iterator())
        svm = LinearSVC(C=best_C)
        svm.fit(train_features, train_targets.ravel())
        s = 1000 * numpy.sum(
            [svm.score(test_features[1000 * i: 1000 * (i + 1)],
                       test_targets[1000 * i: 1000 * (i + 1)].ravel())
             for i in range(26)])
        s += 32 * svm.score(test_features[-32:], test_targets[-32:].ravel())
        s = s / 26032.0
        error_rates.append(1 - s)

    print('Test error rate = {} +- {} '.format(numpy.mean(error_rates),
                                               numpy.std(error_rates)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALI-based semi-supervised "
                                                 "training on SVHN")
    parser.add_argument("dataset_path", type=str,
                        help="path to the saved main loop")
    parser.add_argument("--use-c", type=float, default=None,
                        help="evaluate using a specific C value")
    parser.add_argument("--log-min", type=float, default=-20,
                        help="minimum C value in log-space")
    parser.add_argument("--log-max", type=float, default=20,
                        help="maximum C value in log-space")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="number of values to try")
    args = parser.parse_args()
    main(args.dataset_path, args.use_c, args.log_min, args.log_max,
         args.num_steps)
