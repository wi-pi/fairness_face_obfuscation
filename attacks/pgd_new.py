"""The Projected Gradient Descent attack."""

import numpy as np
import tensorflow as tf

from attacks.fgsm import fast_gradient_method


def clip_eta(eta, norm, eps):
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")
    axis = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("")
        elif norm == 2:
            norm = tf.sqrt(
                tf.maximum(
                    avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)
                )
            )
        factor = tf.minimum(1.0, tf.math.divide(eps, norm))
        eta = eta * factor
    return eta


def random_exponential(shape, rate=1.0, dtype=tf.float32, seed=None):
    return tf.random.gamma(shape, alpha=1, beta=1.0 / rate, dtype=dtype, seed=seed)


def random_laplace(shape, loc=0.0, scale=1.0, dtype=tf.float32, seed=None):
    z1 = random_exponential(shape, 1.0 / scale, dtype=dtype, seed=seed)
    z2 = random_exponential(shape, 1.0 / scale, dtype=dtype, seed=seed)
    return z1 - z2 + loc


def random_lp_vector(shape, ord, eps, dtype=tf.float32, seed=None):
    if ord not in [np.inf, 1, 2]:
        raise ValueError("ord must be np.inf, 1, or 2.")

    if ord == np.inf:
        r = tf.random.uniform(shape, -eps, eps, dtype=dtype, seed=seed)
    else:
        dim = tf.reduce_prod(shape[1:])

        if ord == 1:
            x = random_laplace(
                (shape[0], dim), loc=1.0, scale=1.0, dtype=dtype, seed=seed
            )
            norm = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)
        elif ord == 2:
            x = tf.random.normal((shape[0], dim), dtype=dtype, seed=seed)
            norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
        else:
            raise ValueError("ord must be np.inf, 1, or 2.")

        w = tf.pow(
            tf.random.uniform((shape[0], 1), dtype=dtype, seed=seed),
            1.0 / tf.cast(dim, dtype),
        )
        r = eps * tf.reshape(w * x / norm, shape)

    return r


# TODO
def PGD(params, model_fn, x, src, tar, margin,
                               clip_min=None, clip_max=None,
                               rand_init=None, rand_minmax=None, sanity_checks=False):
    eps = params['epsilon']
    eps_iter = params['epsilon_steps']
    nb_iter = params['iterations']
    norm = int(params['norm'])
    assert eps_iter <= eps, (eps_iter, eps)
    if norm == 1:
        raise NotImplementedError("It's not clear that FGM is a good inner loop"
                              " step for PGD when norm=1, because norm=1 FGM "
                              " changes only one pixel at a time. We need "
                              " to rigorously test a strong norm=1 PGD "
                              "before enabling this feature.")
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    # Initialize loop variables
    if rand_minmax is None:
        rand_minmax = eps

    if rand_init:
        eta = random_lp_vector(tf.shape(x), norm, tf.cast(rand_minmax, x.dtype), dtype=x.dtype)
    else:
        eta = tf.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    boxmin = 0
    boxmax = 1
    
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.

    target_imgs_tanh = tf.tanh(tar)*boxmul + boxplus
    src_imgs_tanh = tf.tanh(src) * boxmul + boxplus

    outputTarg = model_fn(target_imgs_tanh)
    outputSelf = model_fn(src_imgs_tanh)

    i = 0
    while i < nb_iter:
        adv_x, loss = fast_gradient_method(params, model_fn, adv_x, outputSelf, outputTarg, eps_iter, norm, margin, clip_min=clip_min,
                                 clip_max=clip_max)
        print('\rStep {} of {}, loss: {}'.format(i, nb_iter, loss), end='')
        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x
