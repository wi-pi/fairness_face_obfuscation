"""The CarliniWagnerL2 attack.
"""
import numpy as np
import tensorflow as tf
import sys, math
from tqdm import tqdm
import Config

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = False       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGET_FLAG = False          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-2     # the initial constant c to pick as a first guess
MARGIN = 0
TV_FLAG = False
LARGE = 1e10

#needed for l_inf attack
LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
REDUCE_CONST = True    # try to lower c each iteration; faster to set to false
CONST_FACTOR = 4.0      # f>1, rate at which we increase constant, smaller better
DECREASE_FACTOR = 0.5   # 0<f<1, rate at which we shrink tau; larger is more accurate


def clip_tanh(x, clip_min, clip_max):
    return ((tf.tanh(x) + 1) / 2) * (clip_max - clip_min) + clip_min


def l2(x, y):
    # technically squarred l2
    return tf.reduce_sum(tf.square(x - y), list(range(1, len(x.shape))))


def CW(model,
       params,
       imgs,
       src_imgs,
       target_imgs,
       num_base=1,
       num_src=1,
       num_target=1,
       margin=MARGIN,
       abort_early=ABORT_EARLY):
    """
    This is the function interface for the Carlini-Wagner-L2 attack.
    For more details on the attack and the parameters see the corresponding class.
    """
    return CarliniWagnerL2(model, params, src_imgs, target_imgs, num_base, num_src, num_target, margin, abort_early).attack(imgs, src_imgs, target_imgs)


class CarliniWagnerL2Exception(Exception):
    pass


class CarliniWagnerL2(object):
    def __init__(
        self,
        model_fn,
        params,
        src,
        tar,
        num_base=1,
        num_src=1,
        num_target=1,
        margin=MARGIN,
        abort_early=ABORT_EARLY,
        clip_min=0.0,
        clip_max=1.0,
    ):
        """
        This attack was originally proposed by Carlini and Wagner. It is an
        iterative attack that finds adversarial examples on many defenses that
        are robust to other attacks.
        Paper link: https://arxiv.org/abs/1608.04644
        At a high level, this attack is an iterative attack using Adam and
        a specially-chosen loss function to find adversarial examples with
        lower distortion than other attacks. This comes at the cost of speed,
        as this attack is often much slower than others.
        :param model_fn: a callable that takes an input tensor and returns the model logits.
        :param y: (optional) Tensor with target labels.
        :param targeted: (optional) Targeted attack?
        :param batch_size (optional): Number of attacks to run simultaneously.
        :param clip_min: (optional) float. Minimum float values for adversarial example components.
        :param clip_max: (optional) float. Maximum float value for adversarial example components.
        :param binary_search_steps (optional): The number of times we perform binary
                                search to find the optimal tradeoff-
                                constant between norm of the purturbation
                                and confidence of the classification.
        :param max_iterations (optional): The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early (optional): If true, allows early aborts if gradient descent
                        is unable to make progress (i.e., gets stuck in
                        a local minimum).
        :param confidence (optional): Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param initial_const (optional): The initial tradeoff-constant used to tune the
                          relative importance of the size of the perturbation
                          and confidence of classification.
                          If binary_search_steps is large, the initial
                          constant is not important. A smaller value of
                          this constant gives lower distortion results.
        :param learning_rate (optional): The learning rate for the attack algorithm.
                          Smaller values produce better results but are
                          slower to converge.
        """
        self.model_fn = model_fn
        self.TARGET_FLAG = params['targeted_flag']
        self.learning_rate = params['learning_rate']
        self.max_iterations = params['iterations']
        #params['iterations']
        self.binary_search_steps = params['binary_steps']
        self.abort_early = abort_early
        self.MARGIN = margin
        if params['batch_size'] <= 0:
            self.batch_size = num_base
        else:
            self.batch_size = min(params['batch_size'], num_base)
        self.num_target = num_target
        self.num_src = num_src
        self.is_hinge_loss = params['hinge_flag']
        self.p_norm = params['norm']
        if self.p_norm != '2':
            self.batch_size = 1
        self.initial_const = params['init_const']
        self.TV_FLAG = params['tv_flag']
        self.COS_FLAG = params['cos_flag']
        self.LOSS_IMPL = params['mean_loss']
        self.boxmin = params['pixel_min']
        self.boxmax = params['pixel_max']

        self.repeat = self.binary_search_steps >= 10

        print('Batch size: {}'.format(self.batch_size))
        print('Margin: {}'.format(self.MARGIN))

        self.clip_min = clip_min
        self.clip_max = clip_max

        # the optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        boxmin = 0
        boxmax = 1
        
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.

        src = tf.cast(src, tf.float32)
        tar = tf.cast(tar, tf.float32)
        src = tf.atanh((src - self.boxplus) / self.boxmul * 0.999999)
        tar = tf.atanh((tar - self.boxplus) / self.boxmul * 0.999999)
        src_new = tf.tanh(src)
        tar_new = tf.tanh(tar)
        src_new = src_new * self.boxmul + self.boxplus
        tar_new = tar_new * self.boxmul + self.boxplus

        # src = (src - self.clip_min) / (self.clip_max - self.clip_min)
        # tar = (tar - self.clip_min) / (self.clip_max - self.clip_min)
        # src = tf.clip_by_value(src, 0.0, 1.0)
        # tar = tf.clip_by_value(tar, 0.0, 1.0)
        # src = (src * 2.0) - 1.0
        # tar = (tar * 2.0) - 1.0
        # src = tf.atanh(src * 0.999999)
        # tar = tf.atanh(tar * 0.999999)
        # src_new = clip_tanh(src, clip_min=self.clip_min, clip_max=self.clip_max)
        # tar_new = clip_tanh(tar, clip_min=self.clip_min, clip_max=self.clip_max)
        outputSelf = self.model_fn(src_new)
        outputTarg = self.model_fn(tar_new)
        self.outputSelfMean = tf.reduce_mean(outputSelf, axis=0)
        self.outputTargMean = tf.reduce_mean(outputTarg, axis=0)


        super(CarliniWagnerL2, self).__init__()

    def attack(self, x, src, tar):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        lp_list = []
        const_list = []
        adv_list = []
        delta_list = []
        for i in range(0, len(x), self.batch_size):
            self.batch_size = min(self.batch_size, len(x) - i)
            lp, const, adv, delta = self._attack(x[i : min(i + self.batch_size, len(x))], src, tar)
            lp_list.extend(lp)
            const_list.extend(const)
            adv_list.extend(adv)
            delta_list.extend(delta)
        r = np.squeeze(np.array([(lp_list, const_list, adv_list, delta_list)]))
        return r

    def _attack(self, x, src, tar):
        if self.clip_min is not None:
            if not np.all(tf.math.greater_equal(x, self.clip_min)):
                raise CarliniWagnerL2Exception(
                    f"The input is smaller than the minimum value of {self.clip_min}r"
                )

        if self.clip_max is not None:
            if not np.all(tf.math.less_equal(x, self.clip_max)):
                raise CarliniWagnerL2Exception(
                    f"The input is greater than the maximum value of {self.clip_max}!"
                )

        # cast to tensor if provided as numpy array
        original_x = tf.cast(x, tf.float32)

        shape = original_x.shape
        print(shape)

        # re-scale x to [0, 1]
        x = original_x
        # x = (x - self.clip_min) / (self.clip_max - self.clip_min)
        # x = tf.clip_by_value(x, 0.0, 1.0)

        # # scale to [-1, 1]
        # x = (x * 2.0) - 1.0

        # # convert tonh-space
        # x = tf.atanh(x * 0.999999)
        x = tf.atanh((x - self.boxplus) / self.boxmul * 0.999999)


        # parameters for the binary search
        lower_bound = tf.zeros(shape[:1])
        upper_bound = tf.ones(shape[:1]) * 1e10

        const = tf.ones(shape[:1]) * self.initial_const

        # placeholder variables for best values
        best_l2 = tf.fill(shape[:1], 1e10)
        best_attack = original_x
        best_delta = original_x
        best_const = tf.fill(shape[:1], -1)
        best_score = tf.fill(shape[:1], -1)
        best_score = tf.cast(best_score, tf.int32)

        compare_fn = tf.math.greater_equal

        # the perturbation
        # modifier = tf.Variable(tf.zeros(shape, dtype=x.dtype), trainable=True)
        modifier = tf.Variable(tf.random.uniform(shape, minval=-0.1, maxval=0.1, dtype=x.dtype), trainable=True)

        for outer_step in range(self.binary_search_steps):
            # at each iteration reset variable state
            modifier.assign(tf.random.uniform(shape, minval=-0.1, maxval=0.1, dtype=x.dtype))
            for var in self.optimizer.variables():
                var.assign(tf.zeros(var.shape, dtype=var.dtype))

            # variables to keep track in the inner loop
            current_best_l2 = tf.fill(shape[:1], 1e10)
            current_best_score = tf.fill(shape[:1], -1)
            current_best_score = tf.cast(current_best_score, tf.int32)
            current_best_loss = 1e10
            current_best_attack = original_x
            current_best_dist = tf.fill(shape[:1], 1e10)

            # The last iteration (if we run many steps) repeat the search once.
            if (
                self.binary_search_steps >= 10
                and outer_step == self.binary_search_steps - 1
            ):
                const = upper_bound

            # early stopping criteria
            prev = None

            for iteration in range(self.max_iterations):
                x_new, loss, l2_dist, pred = self.attack_step(x, modifier, const)
                print('\rStep {} of {}, loss: {}'.format(iteration, self.max_iterations, loss), end='')
                # print('\rStep {} of {}, loss: {}, dist: {}, l2: {}'.format(iteration, self.max_iterations, loss, pred, l2_dist), end='')
                # print(loss, pred, l2_dist)

                # check if we made progress, abort otherwise
                if (
                    self.abort_early
                    and iteration % ((self.max_iterations // 10) or 1) == 0
                ):
                    if prev is not None and loss > prev * 0.9999:
                        break

                    prev = loss

                temp_attack = []
                temp_dist = []
                temp_l2 = []
                if loss < current_best_loss:
                    for e in range(self.batch_size):
                        temp_attack.append(x_new[e])
                        temp_dist.append(pred[e])
                        temp_l2.append(l2_dist[e])
                        current_best_loss = loss
                    current_best_attack = tf.stack(temp_attack)
                    current_best_dist = tf.stack(temp_dist)
                    current_best_l2 = tf.stack(temp_l2)

            temp_best_attack = []
            temp_best_delta = []
            temp_best_l2 = []
            temp_best_const = []
            temp_upper_bound = []
            temp_lower_bound = []
            temp_const_list = []
            for e in range(self.batch_size):
                if current_best_dist[e] >= self.MARGIN:
                    #success condition, decrease const
                    delta = current_best_attack[e] - original_x[e]
                    if current_best_l2[e] < best_l2[e]:
                        temp_best_attack.append(current_best_attack[e])
                        temp_best_delta.append(delta)
                        temp_best_l2.append(current_best_l2[e])
                        temp_best_const.append(const[e])
                    else:
                        temp_best_attack.append(best_attack[e])
                        temp_best_delta.append(best_delta[e])
                        temp_best_l2.append(best_l2[e])
                        temp_best_const.append(best_const[e])
                    temp_upper_bound.append(min(upper_bound[e], const[e]))
                    temp_lower_bound.append(lower_bound[e])
                    if temp_upper_bound[e] < 1e9:
                        temp_const_list.append((temp_upper_bound[e] + temp_lower_bound[e]) / 2)
                    else:
                        temp_const_list.append(const[e])
                    print('Img: {}, decrease const between {} and {}'.format(e, const[e], temp_const_list[e]))
                else:
                    #failure condition, increase const
                    temp_best_attack.append(best_attack[e])
                    temp_best_delta.append(best_delta[e])
                    temp_best_l2.append(best_l2[e])
                    temp_best_const.append(best_const[e])
                    temp_upper_bound.append(upper_bound[e])
                    temp_lower_bound.append(max(const[e], lower_bound[e]))
                    if temp_upper_bound[e] < 1e9:
                        temp_const_list.append((temp_upper_bound[e] + temp_lower_bound[e]) / 2)
                    else:
                        temp_const_list.append(const[e] * 10)
                    print('Img: {}, increase const between {} and {}'.format(e, const[e], temp_const_list[e]))
            best_attack = tf.stack(temp_best_attack)
            best_delta = tf.stack(temp_best_delta)
            best_l2 = tf.stack(temp_best_l2)
            best_const = tf.stack(temp_best_l2)
            upper_bound = tf.stack(temp_upper_bound)
            lower_bound = tf.stack(temp_lower_bound)
            const = tf.stack(temp_const_list)

        return best_l2, best_const, best_attack, best_delta

    def attack_step(self, x, modifier, const):
        x_new, grads, loss, l2_dist, pred = self.gradient(x, modifier, const)
        self.optimizer.apply_gradients([(grads, modifier)])
        return x_new, loss, l2_dist, pred

    #@tf.function
    def gradient(self, x, modifier, const):
        # compute the actual attack
        with tf.GradientTape() as tape:
            adv_image = modifier + x
            # x_new = clip_tanh(adv_image, clip_min=self.clip_min, clip_max=self.clip_max)
            x_new = tf.tanh(adv_image)
            x_new = x_new * self.boxmul + self.boxplus
            x_new_out = self.model_fn(x_new.numpy())

            loss, l2_dist, pred = self.loss_fn(
                x=x,
                x_new=x_new,
                x_new_out=x_new_out,
                const=const,
            )

        grads = tape.gradient(loss, adv_image)
        return x_new, grads, loss, l2_dist, pred

    def loss_fn(
        self,
        x,
        x_new,
        x_new_out,
        const,
    ):
        # other = clip_tanh(x, clip_min=clip_min, clip_max=clip_max)
        other = tf.tanh(x)
        other = other * self.boxplus + self.boxplus
        l2_dist = tf.sqrt(tf.reduce_sum(tf.square(x_new - other), range(1, len(x.shape))))
        # l2_dist = l2(x_new, other)

        def ZERO():
            return np.asarray(0., dtype=np.dtype('float32'))
        src_loss = tf.sqrt(tf.reduce_sum(tf.square(x_new_out - self.outputSelfMean), [1]))
        target_loss = tf.sqrt(tf.reduce_sum(tf.square(x_new_out - self.outputTargMean), [1]))
        pred = src_loss - target_loss
        hinge_loss = target_loss - src_loss + self.MARGIN
        hinge_loss = tf.maximum(hinge_loss, ZERO())
        loss1 = tf.reduce_sum(const * hinge_loss)
        loss2 = tf.reduce_sum(l2_dist)
        loss = loss1 + loss2
        return loss, l2_dist, pred
