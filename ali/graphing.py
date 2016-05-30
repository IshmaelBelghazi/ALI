""" ALI related graphs """
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import scipy

from ali.datasets import GaussianMixtureDistribution
from ali.utils import as_array


def make_2D_latent_view(valid_data,
                        samples_data,
                        gradients_funs=None,
                        densities_funs=None,
                        epoch=None,
                        save_path=None):
    """
    2D views of the latent and visible spaces
    Parameters
    ----------
    valid_data: dictionary of numpy arrays
        Holds five keys: originals, labels, mu, sigma, encoding, reconstructions
    samples_data: dictionary of numpy arrays
        Holds two keys prior and samples
    gradients_funs: dict of functions
        Holds two keys: latent, for the gradients on the latent space w.r.p to Z and
    visible, for the gradients ob the visible space
    densities_fun: dictionary of functions
        Holds two keys: latent, for the probability density of the latent space, and
    visible, for the probability density on the latent space
    """

    # Creating figure
    fig = plt.figure()
    # Getting Cmap
    cmap = plt.cm.get_cmap('Spectral', 5)
    # Adding visible subplot
    recons_visible_ax = fig.add_subplot(221, aspect='equal')
    # Train data
    recons_visible_ax.scatter(valid_data['originals'][:, 0],
                              valid_data['originals'][:, 1],
                              c=valid_data['labels'],
                              marker='s', label='originals',
                              alpha=0.3, cmap=cmap)

    recons_visible_ax.scatter(valid_data['reconstructions'][:, 0],
                              valid_data['reconstructions'][:, 1],
                              c=valid_data['labels'],
                              marker='x', label='reconstructions',
                              alpha=0.3,
                              cmap=cmap)

    recons_visible_ax.set_title('Visible space. Epoch {}'.format(str(epoch)))
    samples_visible_ax = fig.add_subplot(222, aspect='equal',
                                         sharex=recons_visible_ax,
                                         sharey=recons_visible_ax)

    samples_visible_ax.scatter(valid_data['originals'][:, 0],
                               valid_data['originals'][:, 1],
                               c=valid_data['labels'],
                               marker='s', label='originals',
                               alpha=0.3,
                               cmap=cmap)

    samples_visible_ax.scatter(samples_data['samples'][:, 0],
                               samples_data['samples'][:, 1],
                               marker='o', alpha=0.3, label='samples')
    samples_visible_ax.set_title('Visible space. Epoch {}'.format(str(epoch)))

    # plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
    #            shadow=True, title="Legend", fancybox=True)
    # visible_ax.get_legend()

    # Adding latent subplot
    recons_latent_ax = fig.add_subplot(223, aspect='equal')
    recons_latent_ax.scatter(valid_data['encodings'][:, 0],
                             valid_data['encodings'][:, 1],
                             c=valid_data['labels'],
                             marker='x', label='encodings',
                             alpha=0.3, cmap=cmap)
    recons_latent_ax.set_title('Latent space. Epoch {}'.format(str(epoch)))

    samples_latent_ax = fig.add_subplot(224, aspect='equal',
                                        sharex=recons_latent_ax,
                                        sharey=recons_latent_ax)
    samples_latent_ax.scatter(samples_data['noise'][:, 0],
                              samples_data['noise'][:, 1],
                              marker='o', label='noise',
                              alpha=0.3)
    samples_latent_ax.set_title('Latent space. Epoch {}'.format(str(epoch)))

    # plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
    #            shadow=True, title="Legend", fancybox=True)
    # latent_ax.get_legend()
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, transparent=True, bbox_inches='tight')

















if __name__ == '__main__':
    means = map(lambda x:  as_array(x), [[0, 0],
                                         [1, 1],
                                         [-1, -1],
                                         [1, -1],
                                         [-1, 1]])
    std = 0.01
    variances = [np.eye(2) * std for _ in means]
    priors = [1.0/len(means) for _ in means]

    gaussian_mixture = GaussianMixtureDistribution(means=means,
                                                   variances=variances,
                                                   priors=priors)
    originals, labels = gaussian_mixture.sample(1000)
    reconstructions = originals * np.random.normal(size=originals.shape,
                                                   scale=0.05)
    encodings = np.random.normal(size=(1000, 2))
    train_data = {'originals': originals, 'labels': labels,
                  'encodings': encodings,
                  'reconstructions': reconstructions}
    valid_data = train_data

    noise = np.random.normal(size=(1000, 2))
    samples = np.random.normal(size=(1000, 2), scale=0.3)
    samples_data = {'noise': noise,
                    'samples': samples}

    #make_2D_latent_view(train_data, valid_data, samples_data)
    make_assignement_plots(valid_data)


