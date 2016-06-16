# Adversarially Learned Inference

Code for the Adversarially Learned Inference paper.

## Compiling the paper locally

From the repo's root directory,

``` bash
$ cd papers
$ latexmk --pdf adverarially_learned_inference
```

## Requirements

* [Blocks](https://blocks.readthedocs.org/en/latest/), development version
* [Fuel](https://fuel.readthedocs.org/en/latest/), development version

## Setup

Clone the repository, then install with

``` bash
$ pip install -e ALI
```

## Downloading and converting the datasets

Set up your `~/.fuelrc` file:

``` bash
$ echo "data_path: \"<MY_DATA_PATH>\"" > ~/.fuelrc
```

Go to `<MY_DATA_PATH>`:

``` bash
$ cd <MY_DATA_PATH>
```

Download the CIFAR-10 dataset:

``` bash
$ fuel-download cifar10
$ fuel-convert cifar10
$ fuel-download cifar10 --clear
```

Download the SVHN format 2 dataset:

``` bash
$ fuel-download svhn 2
$ fuel-convert svhn 2
$ fuel-download svhn 2 --clear
```

Download the CelebA dataset:

``` bash
$ fuel-download celeba 64
$ fuel-convert celeba 64
$ fuel-download celeba 64 --clear
```

## Training the models

Make sure you're in the repo's root directory.

### CIFAR-10

``` bash
$ THEANORC=theanorc python experiments/ali_cifar10.py
```

### SVHN

``` bash
$ THEANORC=theanorc python experiments/ali_svhn.py
```

### CelebA

``` bash
$ THEANORC=theanorc python experiments/ali_celeba.py
```

### Toy task

``` bash
$ THEANORC=theanorc python experiments/ali_mixture.py
```

``` bash
$ THEANORC=theanorc python experiments/gan_mixture.py
```

## Evaluating the models

### Samples

``` bash
$ THEANORC=theanorc scripts/sample [main_loop.tar]
```

e.g.

``` bash
$ THEANORC=theanorc scripts/sample ali_cifar10.tar
```

### Interpolations

``` bash
$ THEANORC=theanorc scripts/interpolate [which_dataset] [main_loop.tar]
```

e.g.

``` bash
$ THEANORC=theanorc scripts/interpolate celeba ali_celeba.tar
```

### Reconstructions

``` bash
$ THEANORC=theanorc scripts/reconstruct [which_dataset] [main_loop.tar]
```

e.g.

``` bash
$ THEANORC=theanorc scripts/reconstruct cifar10 ali_cifar10.tar
```

### Semi-supervised learning on SVHN

First, preprocess the SVHN dataset with the learned ALI features:

``` bash
$ THEANORC=theanorc scripts/preprocess_representations [main_loop.tar] [save_path.hdf5]
```

e.g.

``` bash
$ THEANORC=theanorc scripts/preprocess_representations ali_svhn.tar ali_svhn_preprocessed.hdf5
```

Then, launch the semi-supervised script:

``` bash
$ python experiments/semi_supervised_svhn.py ali_svhn.tar [save_path.hdf5]
```

e.g.

``` bash
$ python experiments/semi_supervised_svhn.py ali_svhn_preprocessed.hdf5

[...]
Validation error rate = ... +- ...
Test error rate = ... +- ...
```

### Toy task

``` bash
$ THEANORC=theanorc scripts/generate_mixture_plots [ali_main_loop.tar] [gan_main_loop.tar]
```

e.g.

``` bash
$ THEANORC=theanorc scripts/generate_mixture_plots ali_mixture.tar gan_mixture.tar
```
