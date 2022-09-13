"""The Fast Gradient Method attack."""

import numpy as np
import tensorflow as tf


def optimize_linear(grad, eps, norm=np.inf):
    axis = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = tf.sign(grad)
        # The following line should not change the numerical results. It applies only because
        # `optimal_perturbation` is the output of a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the perturbation has a non-zero derivative.
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif norm == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(
            tf.equal(abs_grad, max_abs_grad), dtype=tf.float32
        )
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif norm == 2:
        square = tf.maximum(
            avoid_zero_div, tf.reduce_sum(tf.square(grad), axis, keepdims=True)
        )
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than norm=1 problem
    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation


# TODO
def loss_fn(params, model_fn, x, src, tar, margin):
    target_imgs = tar
    src_imgs = src
    outputNew = model_fn(x)
    outputTargMean = tf.reduce_mean(tar, axis=0)
    outputSelfMean = tf.reduce_mean(src, axis=0)

    target_loss = tf.sqrt(tf.reduce_sum(tf.square(outputNew - outputTargMean), [1]))
    src_loss = tf.sqrt(tf.reduce_sum(tf.square(outputNew - outputSelfMean), [1]))

    def ZERO():
        return np.asarray(0., dtype=np.dtype('float32'))

    hinge_loss = target_loss - src_loss + margin
    hinge_loss = tf.maximum(hinge_loss, ZERO())
    loss = hinge_loss
    loss = -loss
    return loss


# TODO
@tf.function
def compute_gradient(params, model_fn, x, src, tar, margin):
    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(params, model_fn, x, src, tar, margin)
    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad, loss


# TODO
def fast_gradient_method(
    params,
    model_fn,
    x,
    src,
    tar,
    eps,
    norm,
    margin,
    clip_min=None,
    clip_max=None,
    sanity_checks=False,
):

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    grad, loss = compute_gradient(params, model_fn, x, src, tar, margin)

    optimal_perturbation = optimize_linear(grad, eps, norm)
    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x, loss
