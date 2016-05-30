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
objective function involves **no** explicit reconstruction loop.

<a name="gan_intro"></a>

## Quick introduction to GANs

GAN pits two neural networks against each other: a **generator** network
\\(G(\\mathbf{z})\\), which maps a random source of noise to synthetic samples,
and a **discriminator** network \\(D(\\mathbf{x})\\), which takes either a data
sample or a sample from the generator network and predicts where the sample
comes from (data distribution or model distribution). The discriminator is
trained to make accurate predictions, and the generator is trained to output
samples that fool the discriminator into thinking they came from the data
distribution.

![Generative adversarial networks]({{ site.url }}/assets/gan_simple.svg)

To better understand GANs, it is necessary to describe it using a probabilistic
framework. Two marginal distributions are defined:

* the _data_ marginal \\(q(\\mathbf{x})\\), and
* the _model_ marginal \\(p(\\mathbf{x}) =
  \\int p(\\mathbf{z}) p(\\mathbf{x} \\mid \\mathbf{z}) d\\mathbf{z}\\).

This translates to the following diagram:

![Generative adversarial networks]({{ site.url }}/assets/gan_probabilistic.svg)

The following value function is maximized by the discriminator and minimized by
the generator:

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

The adversarial game played between the discriminator and the generator has the
side effect that the two marginal distributions eventually match, i.e.,
\\(p(\\mathbf{x}) \\sim q(\\mathbf{x})\\).

<a name="learning_inference"></a>

## Learning inference

ALI augments the generator with an inference network \\(q(\\mathbf{z} \\mid
\\mathbf{x})\\). This allows to define two joint distributions:

* the _encoder_ joint \\(q(\\mathbf{x}, \\mathbf{z}) = q(\\mathbf{x})
  q(\\mathbf{z} \\mid \\mathbf{x})\\), and
* the _decoder_ joint \\(p(\\mathbf{x}, \\mathbf{z}) = p(\\mathbf{z})
  p(\\mathbf{x} \\mid \\mathbf{z})\\).

Rather than examining \\(\\mathbf{x}\\) samples marginally, the discriminator
now receives joint pairs \\((\\mathbf{x}, \\mathbf{z})\\) as input and must
predict whether they come from the encoder joint or the decoder joint. Like
before, the generator is trained to fool the discriminator, but this time it
can also learn \\(q(\\mathbf{z} \\mid \\mathbf{x})\\).

![Adversarially learned inference]({{ site.url }}/assets/ali_probabilistic.svg)

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

![CIFAR10 samples]({{ site.url }}/assets/cifar10_samples.png)

### Reconstructions

Odd columns are validation set examples, even columns are their corresponding
reconstructions (e.g., first column contains validation set examples, second
column contains their corresponding reconstruction).

![CIFAR10 reconstructions]({{ site.url }}/assets/cifar10_reconstructions.png)

<a name="svhn"></a>

## SVHN

[SVHN](http://ufldl.stanford.edu/housenumbers/) is a dataset of digit images
obtained from house numbers in Google Street View images. It contains over
600,000 labeled examples.

### Samples

![SVHN samples]({{ site.url }}/assets/svhn_samples.png)

### Reconstructions

Odd columns are validation set examples, even columns are their corresponding
reconstructions (e.g., first column contains validation set examples, second
column contains their corresponding reconstruction).

![SVHN reconstructions]({{ site.url }}/assets/svhn_reconstructions.png)

<a name="celeba"></a>

## CelebA

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a dataset of
celebrity faces with 40 attribute annotations. It contains over 200,000 labeled
examples.

### Samples

![CelebA samples]({{ site.url }}/assets/celeba_samples.png)

### Reconstructions

Odd columns are validation set examples, even columns are their corresponding
reconstructions (e.g., first column contains validation set examples, second
column contains their corresponding reconstruction).

![CelebA reconstructions]({{ site.url }}/assets/celeba_reconstructions.png)
