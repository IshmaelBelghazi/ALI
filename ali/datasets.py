"""Additional dataset classes."""
from __future__ import (division, print_function, )
from collections import OrderedDict
from scipy.stats import multivariate_normal

import numpy as np
import numpy.random as npr

from fuel import config
from fuel.datasets import H5PYDataset, IndexableDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path

from ali.utils import as_array


class TinyILSVRC2012(H5PYDataset):
    """The Tiny ILSVRC2012 Dataset.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' (1,281,167 examples)
        'valid' (50,000 examples), and 'test' (100,000 examples).

    """
    filename = 'ilsvrc2012_tiny.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', False)
        super(TinyILSVRC2012, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)


class GaussianMixture(IndexableDataset):
    """ Toy dataset containing points sampled from a gaussian mixture distribution.

    The dataset contains 3 sources:
    * features
    * label
    * densities

    """
    def __init__(self, num_examples, means=None, variances=None, priors=None,
                 **kwargs):
        rng = kwargs.pop('rng', None)
        if rng is None:
            seed = kwargs.pop('seed', config.default_seed)
            rng = np.random.RandomState(seed)
    
        gaussian_mixture = GaussianMixtureDistribution(means=means,
                                                       variances=variances,
                                                       priors=priors,
                                                       rng=rng)
        self.means = gaussian_mixture.means
        self.variances = gaussian_mixture.variances
        self.priors = gaussian_mixture.priors

        features, labels = gaussian_mixture.sample(nsamples=num_examples)
        densities = gaussian_mixture.pdf(x=features)

        data = OrderedDict([
            ('features', features),
            ('label', labels),
            ('density', densities)
        ])

        super(GaussianMixture, self).__init__(data, **kwargs)


class GaussianMixtureDistribution(object):
    """ Gaussian Mixture Distribution

    Parameters
    ----------
    means : tuple of ndarray.
       Specifies the means for the gaussian components.
    variances : tuple of ndarray.
       Specifies the variances for the gaussian components.
    priors : tuple of ndarray
       Specifies the prior distribution of the components.

    """

    def __init__(self, means=None, variances=None, priors=None, rng=None, seed=None):

        if means is None:
            means = map(lambda x:  10.0 * as_array(x), [[0, 0],
                                                        [1, 1],
                                                        [-1, -1],
                                                        [1, -1],
                                                        [-1, 1]])
        # Number of components
        self.ncomponents = len(means)
        self.dim = means[0].shape[0]
        self.means = means
        # If prior is not specified let prior be flat.
        if priors is None:
            priors = [1.0/self.ncomponents for _ in range(self.ncomponents)]
        self.priors = priors
        # If variances are not specified let variances be identity
        if variances is None:
            variances = [np.eye(self.dim) for _ in range(self.ncomponents)]
        self.variances = variances

        assert len(means) == len(variances), "Mean variances mismatch"
        assert len(variances) == len(priors), "prior mismatch"

        if rng is None:
            rng = npr.RandomState(seed=seed)
        self.rng = rng

    def _sample_prior(self, nsamples):
        return self.rng.choice(a=self.ncomponents,
                               size=(nsamples, ),
                               replace=True,
                               p=self.priors)

    def sample(self, nsamples):
        # Sampling priors
        samples = []
        fathers = self._sample_prior(nsamples=nsamples).tolist()
        for father in fathers:
            samples.append(self._sample_gaussian(self.means[father],
                                                 self.variances[father]))
        return as_array(samples), as_array(fathers)

    def _sample_gaussian(self, mean, variance):
        # sampling unit gaussians
        epsilons = self.rng.normal(size=(self.dim, ))

        return mean + np.linalg.cholesky(variance).dot(epsilons)

    def _gaussian_pdf(self, x, mean, variance):
        return multivariate_normal.pdf(x, mean=mean, cov=variance)

    def pdf(self, x):
        "Evaluates the the probability density function at the given point x"
        pdfs = map(lambda m, v, p: p * self._gaussian_pdf(x, m, v),
                   self.means, self.variances, self.priors)
        return reduce(lambda x, y: x + y, pdfs, 0.0)


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
    gmdset = GaussianMixture(1000, means, variances, priors, sources=('features', ))

