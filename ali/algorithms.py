"""ALI-related training algorithms."""
from collections import OrderedDict

import theano
from blocks.algorithms import GradientDescent, CompositeRule, Restrict


def ali_algorithm(discriminator_loss, discriminator_parameters,
                  discriminator_step_rule, generator_loss,
                  generator_parameters, generator_step_rule):
    """Instantiates a training algorithm for ALI.

    Parameters
    ----------
    discriminator_loss : tensor variable
        Discriminator loss.
    discriminator_parameters : list
        Discriminator parameters.
    discriminator_step_rule : :class:`blocks.algorithms.StepRule`
        Discriminator step rule.
    generator_loss : tensor variable
        Generator loss.
    generator_parameters : list
        Generator parameters.
    generator_step_rule : :class:`blocks.algorithms.StepRule`
        Generator step rule.
    """
    gradients = OrderedDict()
    gradients.update(
        zip(discriminator_parameters,
            theano.grad(discriminator_loss, discriminator_parameters)))
    gradients.update(
        zip(generator_parameters,
            theano.grad(generator_loss, generator_parameters)))
    step_rule = CompositeRule([Restrict(discriminator_step_rule,
                                        discriminator_parameters),
                               Restrict(generator_step_rule,
                                        generator_parameters)])
    return GradientDescent(
        cost=generator_loss + discriminator_loss,
        gradients=gradients,
        parameters=discriminator_parameters + generator_parameters,
        step_rule=step_rule)
