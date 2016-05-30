"""
Visualization of 2D mixture learned with ALI
"""
from collections import OrderedDict
from functools import partial

import matplotlib
print("Using Backend: ", matplotlib.get_backend())

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import theano
from theano import tensor

from blocks.serialization import load
from blocks.bricks.interfaces import Random

import matplotlib.pyplot as plt
from ali.streams import create_gaussian_mixture_data_streams
from ali.utils import (as_array, )

LABELS_CMAP = 'Spectral'
PROB_CMAP = 'jet'
GRADS_GRID_NPTS = 20  # NUmber of points in the gradients grid
NGRADS = 1  # Number of gradients skipped in the quiver plot
SCATTER_ALPHA = 0.5  # Scatter plots transparency
MARKERSIZE = 8

#############
## Helpers ##
#############
def get_key_from_val(dictionary, target_val):
    for key, val in dictionary.items():
        if val == target_val:
            return key
    return None


def get_data(main_loop, n_points=1000):
    means = main_loop.data_stream.dataset.means
    variances = main_loop.data_stream.dataset.variances
    priors = main_loop.data_stream.dataset.priors
    _, _, stream = create_gaussian_mixture_data_streams(n_points, n_points,
                                                        sources=('features',
                                                                 'label'),
                                                        means=means,
                                                        variances=variances,
                                                        priors=priors)
    originals, labels = next(stream.get_epoch_iterator())
    return {'originals': originals,
            'labels': labels}


def get_compiled_functions(main_loop):
    ali, = main_loop.model.top_bricks
    x = tensor.matrix('x')
    z = tensor.matrix('z')

    # Accuracies
    accuracies = ali.get_accuracies(x, z)
    accuracies_fun = theano.function([x, z], accuracies)
    # Encoding decoding
    encoding = ali.sample_z_hat(x)
    encoding_fun = theano.function([x], encoding)
    decoding = ali.sample_x_tilde(z)
    decoding_fun = theano.function([z], decoding)

    # losses and latent gradients
    disc_loss, gen_loss = ali.compute_losses(x, z)
    disc_loss_fun = theano.function([x, z], disc_loss)
    gen_loss_fun = theano.function([x, z], gen_loss)

    disc_loss_z_grads = tensor.grad(disc_loss, z)
    gen_loss_z_grads = tensor.grad(gen_loss, z)

    disc_loss_x_grads = tensor.grad(disc_loss, x)
    gen_loss_x_grads = tensor.grad(gen_loss, x)

    disc_grad_z_fun = theano.function([x, z], disc_loss_z_grads)
    gen_grad_z_fun = theano.function([x, z], gen_loss_z_grads)

    disc_grad_x_fun = theano.function([x, z], disc_loss_x_grads)
    gen_grad_x_fun = theano.function([x, z], gen_loss_x_grads)

    gradients_funs = {'discriminator': {'X_grads': disc_grad_x_fun,
                                        'Z_grads': disc_grad_z_fun},
                      'generator': {'X_grads': gen_grad_x_fun,
                                    'Z_grads': gen_grad_z_fun}}

    return (gradients_funs,
            accuracies_fun,
            {'encode': encoding_fun,
             'decode': decoding_fun})


def mouseevent_to_nparray(event):
    return as_array((event.xdata, event.ydata))

################
## Visualizer ##
################
class MixtureVisualizer(object):
    def __init__(self, main_loop, ngrid_pts):
        self.main_loop = main_loop
        self.ngrid_pts = ngrid_pts
        self.fig, axes = plt.subplots(nrows=2, ncols=3)
        self.axes = OrderedDict(zip(['X', 'Z', 'X_of_Z',
                                     'Info','Z_grads', 'X_grads'], axes.ravel()))
        self.scatter_plots = OrderedDict(zip(self.axes.keys(),
                                             [None] * 6))
        self.grads_plots = OrderedDict(zip(['X_grads', 'Z_grads'],
                                           [None] * 2))
        self.prob_plots = OrderedDict(zip(['Z', 'X_of_Z'],
                                          [None] * 2))
        # getting compiled functions
        comp_funs = get_compiled_functions(main_loop)
        self._get_grads = comp_funs[0]
        self._get_accuracies = comp_funs[1]
        self._get_mappings = comp_funs[2]

        # getting validation data
        self.data = get_data(self.main_loop)
        self.n_classes = len(self.main_loop.data_stream.dataset.priors)

        self.add_titles()
        self.add_scatters()
        selected_x = self.features[0]
        selected_z = self.codes[0]
        # self.update_gradients_field('Z_grads', selected_x)
        # self.update_gradients_field('X_grads', selected_z)

        # Adding initial probability Maps
        self.selected_id = {'X': {'base': None,
                                  'target_prob': None,
                                  'target_grad': None},
                            'Z': {'base': None,
                                  'target_prob': None,
                                  'target_grad': None}}

        self.update_probability_map('Z', self.features[0])
        self.update_probability_map('X_of_Z', self.codes[0])
        self.finetune_axes()
        self.register_callbacks()

    @property
    def labels(self):
        return self.data['labels']

    @property
    def features(self):
        return self.data['originals']

    @property
    def codes(self):
        return self._get_mappings['encode'](self.features)

    @property
    def reconstructions(self):
        return self._get_mappings['decode'](self.codes)

    @property
    def current_epoch(self):
        return self.main_loop.status['epochs_done']

    def add_scatter(self, name, datum, label):
        ax = self.axes[name].scatter(*(self._split_arr(datum)),
                                     c=self.labels,
                                     s=50,
                                     marker='o',
                                     label=label,
                                     alpha=SCATTER_ALPHA,
                                     cmap=plt.cm.get_cmap(LABELS_CMAP,
                                                          self.n_classes))
        self.scatter_plots[name] = ax

    def add_scatters(self):
        names = ['X',  'Z', 'X_of_Z', 'X_grads', 'Z_grads']
        data = [self.features, self.codes, self.reconstructions,
                self.reconstructions, self.codes]
        labels = ['originals', 'encodings', 'reconstructions',
                  'reconstructions', 'encodings']
        assert len(names) == len(data) == len(labels)

        for name, datum, label in zip(names, data, labels):
            self.add_scatter(name=name, datum=datum, label=label)

    def add_probability_map(self, name, accuracies):
        im = self.axes[name].imshow(accuracies,
                                    cmap=plt.cm.get_cmap(PROB_CMAP),
                                    extent=self._get_extent(name),
                                    vmin=0.0, vmax=1.0)
        self.prob_plots[name] = im

    def update_probability_map(self, name, selected):
        accuracies = self.get_accuracies(name, selected)
        # Annoying Qt4 bug forces redrawing the entire image,
        self.add_probability_map(name, accuracies)

    def update_gradients_field(self, name, selected):
        # Getting gradients and grid
        grad_x, grad_y, x, y = self.get_gradients(name, selected)
        # plot every n grads
        if self.grads_plots[name] is None:
            quiv = self.axes[name].quiver(x[::NGRADS], y[::NGRADS],
                                          grad_x[::NGRADS], grad_y[::NGRADS])
            self.grads_plots[name] = quiv
        else:
            # assert self.grads_plots[name] is not None
            self.grads_plots[name].set_UVC(grad_x[::NGRADS], grad_y[::NGRADS])

    def add_titles(self):
        self.fig.suptitle('ALI - Gaussian Mixture - Epoch: {}'.format(
            self.current_epoch)
        )

        self.axes['X'].set_title('Validation')
        self.axes['Z'].set_title('Validation Encodings & Data Accuracies')
        self.axes['X_of_Z'].set_title('Reconstructions & Sample Accuracies')
        self.axes['X_grads'].set_title('Discriminator score w.r.p to x')
        self.axes['Z_grads'].set_title('Discriminator score w.r.p to z')

    def finetune_axes(self):
        # Forcing subplots to have 'box' aspect
        for ax in self.axes.values():
            ax.set_aspect('equal', adjustable='box')
            ax.set_autoscale_on(False)

        # Setting ylim and xlim
        X_axes = ['X_grads', 'X_of_Z']
        for ax_name in X_axes:
            self.axes[ax_name].set_xlim(self.axes['X'].get_xlim())
            self.axes[ax_name].set_ylim(self.axes['X'].get_ylim())
        self.axes['Z_grads'].set_xlim(self.axes['Z'].get_xlim())
        self.axes['Z_grads'].set_ylim(self.axes['Z'].get_ylim())

        # Adding colorbar
        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
        self.fig.colorbar(self.prob_plots['Z'], cax=cbar_ax)

    def get_grid(self, ax, num=None):
        if num is None:
            num = self.ngrid_pts
        x = np.linspace(*ax.get_xlim(), num=num)
        y = np.linspace(*ax.get_ylim(), num=num)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    def get_accuracies(self, name, selected):
        assert name in ['X_of_Z', 'Z']
        xx, yy = self.get_grid(self.axes[name])
        grid = np.vstack([xx.flatten(order='F'), yy.flatten(order='F')]).T
        selected_grid = np.tile(selected, (grid.shape[0], 1))
        if name == 'X_of_Z':
            input_grids = [grid, selected_grid]
        elif name == 'Z':
            input_grids = [selected_grid, grid]
        accuracies = self._get_accuracies(*input_grids).reshape(xx.shape,
                                                                order='F')
        return accuracies

    def get_Z_gradients(self, selected_x):
        xx, yy = self.get_grid(self.axes['Z_grads'], GRADS_GRID_NPTS)
        assert xx.shape == yy.shape
        grads_shape = xx.shape
        grid = np.vstack([xx.flatten(order='F'), yy.flatten(order='F')]).T
        x0 = np.tile(selected_x, (grid.shape[0], 1))
        grads = self._get_grads['discriminator']['Z_grads'](x0, grid)
        return [grad.reshape(grads_shape, order='F')
                for grad in self._split_arr(grads)] + [xx, yy]

    def get_X_gradients(self, selected_z):
        xx, yy = self.get_grid(self.axes['X_grads'], GRADS_GRID_NPTS)
        assert xx.shape == yy.shape
        grads_shape = xx.shape
        grid = np.vstack([xx.flatten(order='F'), yy.flatten(order='F')]).T
        z0 = np.tile(selected_z, (grid.shape[0], 1))
        grads = self._get_grads['discriminator']['X_grads'](grid, z0)
        return [grad.reshape(grads_shape, order='F')
                for grad in self._split_arr(grads)] + [xx, yy]

    def get_gradients(self, name, selected):
        assert name in ['X_grads', 'Z_grads']
        if name == 'X_grads':
            return self.get_X_gradients(selected)
        elif name == 'Z_grads':
            return self.get_Z_gradients(selected)

    def remove_previously_selected(self, name):
        for renderer in self.selected_id[name].values():
            if renderer is not None:
                renderer.remove()

    def mark_selected(self, name, prob_target_name, grad_target_name, selected_val):
        mapping_name = 'encode' if name == 'X' else 'decode'
        mapped_val = self._get_mappings[mapping_name](
            selected_val.reshape(1, selected_val.shape[0])).flatten()
        marker_style = '^r' if name == 'X' else '^b'

        # Adding selected val
        self.selected_id[name]['base'], = self.axes[name].plot(
            selected_val[0], selected_val[1],
            marker_style, markersize=MARKERSIZE)

        # Adding mapped val
        self.selected_id[name]['prob_target'], = self.axes[prob_target_name].plot(
            mapped_val[0], mapped_val[1],
            marker_style, markersize=MARKERSIZE
        )
        self.selected_id[name]['grad_target'], = self.axes[grad_target_name].plot(
            mapped_val[0], mapped_val[1],
            marker_style, markersize=MARKERSIZE
        )

    def click_event(self, event):
        # Getting current ax
        inax = event.inaxes
        # get current ax identity
        ax_name = get_key_from_val(self.axes, inax)

        isvalid_axis = ax_name in ['X', 'Z']
        isvalid_pt = event.xdata is not None and event.ydata is not None
        if isvalid_axis and isvalid_pt:
            selected_val = mouseevent_to_nparray(event)
            prob_target_name = 'Z' if ax_name == 'X' else 'X_of_Z'
            self.update_probability_map(prob_target_name,
                                        selected_val)

            grad_target_name = 'Z_grads' if ax_name == 'X' else 'X_grads'
            self.update_gradients_field(grad_target_name, selected_val)

            self.remove_previously_selected(ax_name)
            self.mark_selected(ax_name, prob_target_name, grad_target_name,
                               selected_val)
            # Updating figure
            plt.pause(0.0001)
            # self.fig.canvas.draw()

    def register_callbacks(self):
        self.fig.canvas.mpl_connect('button_press_event', self.click_event)

    def _split_arr(self, arr):
        return np.split(arr, 2, axis=1)

    def _get_extent(self, axis_name):
        "Returns (xmin, xmax, ymin, ymax)"
        self.axes['Z_grads'].set_ylim(self.axes['Z'].get_xlim())
        return self.axes[axis_name].get_xlim() \
            + self.axes[axis_name].get_ylim()

    def show(self):
        self.fig.show()

if __name__ == '__main__':
    main_loop_path = "../experiments/ali_gm.tar"
    with open(main_loop_path, 'rb') as ali_src:
        main_loop = load(ali_src)

    # Initializing visualizer
    plt.ion()
    ngrid_pts = 200
    mixture_viz = MixtureVisualizer(main_loop, ngrid_pts=ngrid_pts)
    #mixture_viz.show()
