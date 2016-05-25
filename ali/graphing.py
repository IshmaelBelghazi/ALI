""" ALI related graphs """

import numpy as np
import matplotlib.pyplot as plt

from ali.datasets import GaussianMixtureDistribution
from ali.utils import as_array



def make_2D_latent_view(valid_data, samples_data, gradients_funs=None, densities_funs=None,
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
    # Adding visible subplot
    visible_ax = fig.add_subplot(211)
    # Train data
    visible_ax.scatter(valid_data['originals'][:, 0],
                       valid_data['originals'][:, 1],
                       c=valid_data['labels'],
                       marker='s', label='Valid')

    visible_ax.scatter(valid_data['reconstructions'][:, 0],
                       valid_data['reconstructions'][:, 1],
                       c=valid_data['labels'],
                       marker='x', label='Valid')

    visible_ax.scatter(samples_data['samples'][:, 0],
                       samples_data['samples'][:, 1],
                       marker='o')


    # Adding latent subplot
    latent_ax = fig.add_subplot(212)
    latent_ax.scatter(valid_data['encodings'][:, 0],
                      valid_data['encodings'][:, 1],
                      c=valid_data['labels'],
                      marker='x', label='Valid')

    latent_ax.scatter(samples_data['samples'][:, 0],
                      samples_data['samples'][:, 1],
                      marker='o')

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

    make_2D_latent_view(train_data, valid_data, samples_data)



