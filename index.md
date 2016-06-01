---
layout: default
title: {{ site.name }}
---

---

# Table of contents

* [What is ALI?](#what_is)
    * [Quick introduction to GANs](#gan_intro)
    * [Learning inference](#learning_inference)
* [Experimental results](#experimental_results)
    * [CIFAR10](#cifar10)
    * [SVHN](#svhn)
    * [CelebA](#celeba)

---

<a name="what_is"></a>

# What is ALI?

The adversarially learned inference (ALI) model is a deep directed generative
model which jointly learns a generation network and an inference network using
an adversarial process. This model constitutes a novel approach to integrating
efficient inference with the generative adversarial networks (GAN) framework.

What makes ALI unique is that unlike other approaches to learning inference in
deep directed generative models like variational autoencoders (VAEs), the
objective function involves **no** explicit reconstruction loop. This produces
reconstructions that are quite different, in that they do not attempt to be
pixel perfect (leading to blurring), but rather attempt to capture semantically
salient information.

<a name="gan_intro"></a>

## Quick introduction to GANs

GAN pits two neural networks against each other: a **generator** network
\\(G(\\mathbf{z})\\), and a **discriminator** network \\(D(\\mathbf{x})\\).

The generator tries to mimic examples from a training dataset, which is sampled
from the true data distribution \\(q(\\mathbf{x})\\). It does so by transforming
a random source of noise received as input into a synthetic sample.

The discriminator receives a sample, but it is not told where the sample comes
from. Its job is to predict whether it is a data sample or a synthetic sample.

The discriminator is trained to make accurate predictions, and the generator is
trained to output samples that fool the discriminator into thinking they came
from the data distribution.

![Generative adversarial networks]({{ site.baseurl }}/assets/gan_simple.svg)

To better understand GANs, it is necessary to describe it using a probabilistic
framework. Two marginal distributions are defined:

* the _data_ marginal \\(q(\\mathbf{x})\\), and
* the _model_ marginal \\(p(\\mathbf{x}) =
  \\int p(\\mathbf{z}) p(\\mathbf{x} \\mid \\mathbf{z}) d\\mathbf{z}\\).

The generator operates by sampling \\(\\mathbf{z} \\sim p(\\mathbf{z})\\) and
then sampling \\(\\mathbf{x} \\sim p(\\mathbf{x} \\mid \\mathbf{z})\\).
The discriminator receives either the generator sample or
\\(\\mathbf{x} \\sim q(\\mathbf{x})\\) and outputs the probability that its
input was sampled from \\(q(\\mathbf{x})\\).

![Generative adversarial networks]({{ site.baseurl }}/assets/gan_probabilistic.svg)

The adversarial game played between the discriminator and the generator is
formalized by the following value function:

$$
\begin{split}
    \min_G \max_D V(D, G)
    &= \mathbb{E}_{q(\mathbf{x})} [\log(D(\mathbf{x}))] +
       \mathbb{E}_{p(\mathbf{z})} [\log(1 - D(G(\mathbf{z})))] \\
    &= \int q(\mathbf{x}) \log(D(\mathbf{x})) d\mathbf{x} +
       \iint p(\mathbf{z}) p(\mathbf{x} \mid \mathbf{z})
             \log(1 - D(\mathbf{x})) d\mathbf{x} d\mathbf{z}
\end{split}
$$

On one hand, the discriminator is trained to maximize the probability of correctly
classifying data samples and synthetic samples. On the other hand, the
generator is trained to produce samples that fool the discriminator, i.e.
that are unlikely to be synthetic according to the discriminator.

It can be shown that for a fixed generator, the optimal discriminator is

$$
    D^*(\mathbf{x}) = \frac{q(\mathbf{x})}{q(\mathbf{x}) + p(\mathbf{x})}
$$

and that given an optimal discriminator, minimizing the value function with
respect to the generator parameters is equivalent to minimizing the
Jensen-Shannon divergence between \\(p(\\mathbf{x})\\) and \\(q(\\mathbf{x})\\).

In other words, as training progresses, the generator produces synthetic samples
that look more and more like the training data.

<a name="learning_inference"></a>

## Learning inference

Even though GANs are pretty good at producing realistic-looking synthetic
samples, they lack something very important: the ability to do inference.

Inference can loosely be defined as the answer to the following question:

> Given \\(\\mathbf{x}\\), what \\(\\mathbf{z}\\) is likely to have produced it?

This question is exactly what ALI is equipped to answer.

ALI augments GAN's generator with an additional network. This network receives
a data sample as input and produces a synthetic \\(\\mathbf{z}\\) as output.

Expressed in probabilistic terms, ALI defines two joint distributions:

* the _encoder_ joint \\(q(\\mathbf{x}, \\mathbf{z}) = q(\\mathbf{x})
  q(\\mathbf{z} \\mid \\mathbf{x})\\), and
* the _decoder_ joint \\(p(\\mathbf{x}, \\mathbf{z}) = p(\\mathbf{z})
  p(\\mathbf{x} \\mid \\mathbf{z})\\).

ALI also modifies the discriminator's goal. Rather than examining
\\(\\mathbf{x}\\) samples marginally, it now receives joint pairs
\\((\\mathbf{x}, \\mathbf{z})\\) as input and must predict whether they come
from the encoder joint or the decoder joint.

![Adversarially learned inference]({{ site.baseurl }}/assets/ali_probabilistic.svg)

Like before, the generator is trained to fool the discriminator, but this time
it can also learn \\(q(\\mathbf{z} \\mid \\mathbf{x})\\).

In analogy to GAN, the adversarial game played between the discriminator and the
generator has the side effect that the two _joint_ distributions eventually match,
i.e., \\(p(\\mathbf{x}, \\mathbf{z}) \\sim q(\\mathbf{x}, \\mathbf{z})\\). This
also means that the marginals match, and that the conditional on one
distribution is the posterior of the other distribution, and vice versa.

<a name="experimental_results"></a>

# Experimental results

<a name="cifar10"></a>

## CIFAR10

The [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset contains
60,000 32x32 colour images in 10 classes.

### Samples

![CIFAR10 samples]({{ site.baseurl }}/assets/cifar10_samples.png)

### Reconstructions

Odd columns are validation set examples, even columns are their corresponding
reconstructions (e.g., first column contains validation set examples, second
column contains their corresponding reconstruction).

![CIFAR10 reconstructions]({{ site.baseurl }}/assets/cifar10_reconstructions.png)

<a name="svhn"></a>

## SVHN

[SVHN](http://ufldl.stanford.edu/housenumbers/) is a dataset of digit images
obtained from house numbers in Google Street View images. It contains over
600,000 labeled examples.

### Samples

![SVHN samples]({{ site.baseurl }}/assets/svhn_samples.png)

### Reconstructions

Odd columns are validation set examples, even columns are their corresponding
reconstructions (e.g., first column contains validation set examples, second
column contains their corresponding reconstruction).

![SVHN reconstructions]({{ site.baseurl }}/assets/svhn_reconstructions.png)

<a name="celeba"></a>

## CelebA

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a dataset of
celebrity faces with 40 attribute annotations. It contains over 200,000 labeled
examples.

### Samples

![CelebA samples]({{ site.baseurl }}/assets/celeba_samples.png)

### Reconstructions

Odd columns are validation set examples, even columns are their corresponding
reconstructions (e.g., first column contains validation set examples, second
column contains their corresponding reconstruction).

![CelebA reconstructions]({{ site.baseurl }}/assets/celeba_reconstructions.png)
