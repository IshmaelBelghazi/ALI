"""Additional dataset classes."""
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path


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
