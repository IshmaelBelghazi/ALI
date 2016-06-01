#!/usr/bin/env python
import argparse

import theano
from blocks.serialization import load
from matplotlib import cm, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid


def main(main_loop, nrows, ncols, save_path=None):
    ali, = main_loop.model.top_bricks
    input_shape = ali.encoder.get_dim('output')
    z = ali.theano_rng.normal(size=(nrows * ncols,) + input_shape)
    x = ali.sample(z)
    samples = theano.function([], x)()

    figure = pyplot.figure()
    grid = ImageGrid(figure, 111, (nrows, ncols), axes_pad=0.1)

    for sample, axis in zip(samples, grid):
        axis.imshow(sample.transpose(1, 2, 0).squeeze(),
                    cmap=cm.Greys_r, interpolation='nearest')
        axis.set_yticklabels(['' for _ in range(sample.shape[1])])
        axis.set_xticklabels(['' for _ in range(sample.shape[2])])
        axis.axis('off')

    if save_path is None:
        pyplot.show()
    else:
        pyplot.savefig(save_path, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot samples.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    parser.add_argument("--nrows", type=int, default=10,
                        help="number of rows of samples to display.")
    parser.add_argument("--ncols", type=int, default=10,
                        help="number of columns of samples to display.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the generated samples.")
    args = parser.parse_args()

    with open(args.main_loop_path, 'rb') as src:
        main(load(src), args.nrows, args.ncols, args.save_path)
