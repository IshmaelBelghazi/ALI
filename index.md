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
    * [Tiny ImageNet](#tiny_imagenet)
    * [Latent space interpolations](#interpolations)
    * [Conditional generation](#conditional)
    * [Semi-supervised learning](#semi_supervised)
* [Conclusion](#conclusion)

---

**Note: Jeff Donahue, Philipp Krähenbühl and Trevor Darrell at Berkeley
published a paper independently from us on the same idea, which they call
Bidirectional GAN, or BiGAN. You should also check out their awesome work at
[https://arxiv.org/abs/1605.09782](https://arxiv.org/abs/1605.09782).**

<a name="what_is"></a>

# What is ALI?

The adversarially learned inference (ALI) model is a deep directed generative
model which jointly learns a generation network and an inference network using
an adversarial process. This model constitutes a novel approach to integrating
efficient inference with the generative adversarial networks (GAN) framework.

What makes ALI unique is that unlike other approaches to learning inference in
deep directed generative models (like variational autoencoders (VAEs)), the
objective function involves **no** explicit reconstruction loop.  Instead of
focusing on achieving a pixel-perfect reconstruction, ALI tends to produce
believable reconstructions with interesting variations, albeit at the expense
of making some mistakes in capturing exact object placement, color, style and
(in extreme cases) object identity. This is a good thing, because 1) capacity is
not wasted to model trivial factors of variation in the input, and 2) the
learned features are more or less invariant to these trivial factors of
variation, which is what is expected of good feature learning.

These strenghts are showcased via the semi-supervised learning tasks on SVHN and
CIFAR10, where ALI achieves a performance competitive with state-of-the-art.

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

<center><em>GAN: general idea</em></center>

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

<center><em>GAN: probabilistic view</em></center>

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

Like before, the generator is trained to fool the discriminator, but this time
it can also learn \\(q(\\mathbf{z} \\mid \\mathbf{x})\\).

![Adversarially learned inference]({{ site.baseurl }}/assets/ali_probabilistic.svg)

<center><em>ALI: probabilistic view</em></center>

The adversarial game played between the discriminator and the generator is
formalized by the following value function:

$$
\begin{split}
    \min_G \max_D V(D, G)
    &= \mathbb{E}_{q(\mathbf{x})} [\log(D(\mathbf{x}, G_z(\mathbf{x})))] +
       \mathbb{E}_{p(\mathbf{z})} [\log(1 - D(G_x(\mathbf{z}), \mathbf{z}))] \\
    &= \iint q(\mathbf{x}) q(\mathbf{z} \mid \mathbf{x})
             \log(D(\mathbf{x}, \mathbf{z})) d\mathbf{x} d\mathbf{z} \\
    &+ \iint p(\mathbf{z}) p(\mathbf{x} \mid \mathbf{z})
             \log(1 - D(\mathbf{x}, \mathbf{z})) d\mathbf{x} d\mathbf{z}
\end{split}
$$


In analogy to GAN, it can be shown that for a fixed generator, the optimal
discriminator is

$$
    D^*(\mathbf{x}, \mathbf{z}) = \frac{q(\mathbf{x}, \mathbf{z})}
                                       {q(\mathbf{x}, \mathbf{z}) +
                                        p(\mathbf{x}, \mathbf{z})}
$$

and that given an optimal discriminator, minimizing the value function with
respect to the generator parameters is equivalent to minimizing the
Jensen-Shannon divergence between \\(p(\\mathbf{x}, \\mathbf{z})\\) and
\\(q(\\mathbf{x}, \\mathbf{z})\\).

Matching the joints also has the effect of matching the marginals
(i.e., \\(p(\\mathbf{x}) \\sim q(\\mathbf{x})\\) and
\\(p(\\mathbf{z}) \\sim q(\\mathbf{z})\\)) as well as the conditionals /
posteriors (i.e.,
\\(p(\\mathbf{z} \\mid \\mathbf{x}) \\sim q(\\mathbf{z} \\mid \\mathbf{x})\\) and
\\(q(\\mathbf{x} \\mid \\mathbf{z}) \\sim p(\\mathbf{x} \\mid \\mathbf{z})\\)).

<a name="experimental_results"></a>

# Experimental results

**Regarding reconstructions: odd columns are validation set examples, even
columns are their corresponding reconstructions (e.g., first column contains
validation set examples, second column contains their corresponding
reconstruction).**

<a name="cifar10"></a>

## CIFAR10

The [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset contains
60,000 32x32 colour images in 10 classes.

<table>
  <tr>
    <td><img src="{{ site.baseurl }}/assets/cifar10_samples.png"></td>
    <td><img src="{{ site.baseurl }}/assets/cifar10_reconstructions.png"></td>
  </tr>
  <tr>
    <td><center>Samples</center></td>
    <td><center>Reconstructions</center></td>
  </tr>
</table>

<a name="svhn"></a>

## SVHN

[SVHN](http://ufldl.stanford.edu/housenumbers/) is a dataset of digit images
obtained from house numbers in Google Street View images. It contains over
600,000 labeled examples.

<table>
  <tr>
    <td><img src="{{ site.baseurl }}/assets/svhn_samples.png"></td>
    <td><img src="{{ site.baseurl }}/assets/svhn_reconstructions.png"></td>
  </tr>
  <tr>
    <td><center>Samples</center></td>
    <td><center>Reconstructions</center></td>
  </tr>
</table>

<a name="celeba"></a>

## CelebA

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a dataset of
celebrity faces with 40 attribute annotations. It contains over 200,000 labeled
examples.

<table>
  <tr>
    <td><img src="{{ site.baseurl }}/assets/celeba_samples.png"></td>
    <td><img src="{{ site.baseurl }}/assets/celeba_reconstructions.png"></td>
  </tr>
  <tr>
    <td><center>Samples</center></td>
    <td><center>Reconstructions</center></td>
  </tr>
</table>

<a name="tiny_imagenet"></a>

## Tiny ImageNet

The Tiny Imagenet dataset is a version of the
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/) dataset that has
been center-cropped and downsampled to \\(64 \\times 64\\) pixels. It contains
over 1,200,000 labeled examples.

<table>
  <tr>
    <td><img src="{{ site.baseurl }}/assets/tiny_imagenet_samples.png"></td>
    <td><img src="{{ site.baseurl }}/assets/tiny_imagenet_reconstructions.png"></td>
  </tr>
  <tr>
    <td><center>Samples</center></td>
    <td><center>Reconstructions</center></td>
  </tr>
</table>

<a name="interpolations"></a>

## Latent space interpolations

As a sanity check for overfitting, we look at latent space interpolations
between CelebA validation set examples. We sample pairs of validation set
examples \\(\\mathbf{x}\_1\\) and \\(\\mathbf{x}\_2\\) and project them into
\\(\\mathbf{z}\_1\\) and \\(\\mathbf{z}\_2\\) by sampling from the encoder. We
then linearly interpolate between \\(\\mathbf{z}\_1\\) and \\(\\mathbf{z}\_2\\)
and pass the intermediary points through the decoder to plot the input-space
interpolations.

![Latent space interpolations]({{ site.baseurl }}/assets/celeba_interpolations.png)

We observe smooth transitions between pairs of example, and intermediary images
remain believable. This is an indicator that ALI is not concentrating its
probability mass exclusively around training examples, but rather has learned
latent features that generalize well.

<a name="conditional"></a>

## Conditional generation

ALI can also be used for conditional generative modeling by providing the
encoder, decoder and discriminator networks with a conditioning variable
\\(\\mathbf{y}\\).

We apply the conditional version of ALI to CelebA using the dataset’s 40 binary
attributes. The attributes are linearly embedded in the encoder, decoder and
discriminator. We observe how a single element of the latent space z changes
with respect to variations in the attributes vector \\(\\mathbf{y}\\).

![Conditional generation]({{ site.baseurl }}/assets/celeba_conditional_sequence.png)

The row attributes are

* I: male, attractive, young
* II: male, attractive, older
* III: female, attractive, young
* IV: female, attractive, older

The column attributes are

* a: unchanged
* b: black hair
* c: brown hair
* d: blond hair
* e: black hair, wavy hair
* f: blond hair, bangs
* g: blond hair, receding hairline
* h: blond hair, balding
* i: black hair, smiling
* j: black hair, smiling, mouth slightly open
* k: black hair, smiling, mouth slightly open, eyeglasses
* l: black hair, smiling, mouth slightly open, eyeglasses, wearing hat

<a name="semi_supervised"></a>

## Semi-supervised learning

ALI achieves a performance competitive with state-of-the-art on the
semi-supervised SVHN and CIFAR10 tasks. We adapt the discriminative model
proposed in Salimans et al. (2016) [^1] to take both x and z as input.

### SVHN

| Number of labeled examples          | 1000           |
| ----------------------------------- | -------------- |
| VAE (M1 + M2) [^1]                  | 36.02          |
| SWWAE with dropout [^2]             | 23.56          |
| DCGAN + L2-SVM [^3]                 | 22.18          |
| SDGM [^4]                           | 16.61          |
| **GAN (feature matching)** [^7]     | **8.11 ± 1.3** |
| **ALI (ours, no feature matching)** | **7.3**        |

### CIFAR10

| Number of labeled examples          | 1000             | 2000             | 4000             | 8000             |
| ----------------------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Ladder network [^5]                 |                  |                  | 20.40            |                  |
| CatGAN [^6]                         |                  |                  | 19.58            |                  |
| **GAN (feature matching)** [^7]     | **21.83 ± 2.01** | **19.61 ± 2.09** | **18.63 ± 2.32** | **17.72 ± 1.82** |
| **ALI (ours, no feature matching)** | **20.88**        | **20.28**        | **18.3**         | **17.77**        |

Interestingly, Salimans et al. (2016) found that they required an alternative
training strategy for the generator where it tries to match first-order
statistics in the discriminator’s intermediate activations with respect to the
data distribution (they refer to this as feature matching). We found that ALI
did not require feature matching to achieve comparable results.

We are still investigating the differences between ALI and GAN with respect to
feature matching, but we conjecture that the latent representation learned by
ALI is better untangled with respect to the classification task and that it
generalizes better.

<a name="conclusion"></a>

# Conclusion

The adversarially learned inference (ALI) model jointly learns a generation
network and an inference network using an adversarial process. The model learns
mutually coherent inference and generation networks, as exhibited by its
reconstructions. The induced latent variable mapping is shown to be useful,
achieving competitive results on semi-supervised SVHN house number
classification.

---

[^1]: Kingma, D. P., Mohamed, S., Rezende, D. J., and Welling, M. (2014). Semi-supervised learning with deep generative models. In _Advances in Neural Information Processing Systems_, pages 3581–3589.
[^2]: Zhao, J., Mathieu, M., Goroshin, R., and Lecun, Y. (2015). Stacked what-where auto-encoders. _arXiv preprint arXiv:1506.02351_.
[^3]: Radford, A., Metz, L., and Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. _arXiv preprint arXiv:1511.06434_.
[^4]: Maaløe, L., Sønderby, C. K., Sønderby, S. K., and Winther, O. (2016). Auxiliary deep generative models. _arXiv preprint arXiv:1602.05473_.
[^5]: Rasmus, A., Valpola, H., Honkala, M., Berglund, M., and Raiko, T. (2015). Semi-supervised learning with ladder network. In _Advances in Neural Information
Processing Systems_.
[^6]: Springenberg, J. T. (2015). Unsupervised and semi-supervised learning with categorical generative adversarial networks. _arXiv preprint arXiv:1511.06390_.
[^7]: Salimans, T., Goodfellow, I. J., Zaremba, W., Cheung, V., Radford, A., and Chen, X. (2016). Improved techniques for training GANs. _arXiv preprint
  arXiv:1606.03498_.
