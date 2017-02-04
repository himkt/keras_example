import math

import numpy

from chainer import cuda
from chainer import optimizer


class Eve(optimizer.GradientMethod):

    """Eve optimization algorithm.
    See: https://arxiv.org/abs/1611.01505
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, beta3=0.999,
                 eps=1e-8, k=0.1, K=10.):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.eps = eps
        self.k = k
        self.K = K

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['m'] = xp.zeros_like(param.data)
            state['v'] = xp.zeros_like(param.data)
            state['d'] = xp.zeros(1, dtype=param.data.dtype)
            state['f'] = xp.zeros(1, dtype=param.data.dtype)

    def update_one_cpu(self, param, state):
        m, v = state['m'], state['v']
        d, f = state['d'], state['f']

        grad = param.grad

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)

        if self.t > 1:
            old_f = float(cuda.to_cpu(state['f']))
            if self.loss >= old_f:
                delta_t = self.k + 1.
                Delta_t = self.K + 1.
            else:
                delta_t = 1. / (self.K + 1.)
                Delta_t = 1. / (self.k + 1.)

            c = min(max(delta_t, self.loss / old_f), Delta_t)
            new_f = c * f

            r = abs(new_f - old_f) / min(new_f, old_f)
            # d = self.beta3 * d + (1 - self.beta3) * r
            # above statement caused low performance
            # because: loss of significant digits?
            d += (1 - self.beta3) * (r - d)
            f[:] = new_f

        else:
            f[:] = self.loss
            d[:] = 1

        # update
        param.data -= self.lr * m / d * (numpy.sqrt(v) + self.eps)

    """
    TODO: to implement update_one_gpu, it needs to separate computations
    of \hat{f_{t-1}} and d_t from update_one_cpu and create _update_d_f
    """
    # def update_one_gpu(self, param, state):
    #     cuda.elementwise(
    #         'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps',
    #         'T param, T m, T v',
    #         '''m += one_minus_beta1 * (grad - m);
    #            v += one_minus_beta2 * (grad * grad - v);
    #            param -= lr * m / (sqrt(v) + eps);''',
    #         'adam')(param.grad, self.lr, 1 - self.beta1, 1 - self.beta2,
    #                 self.eps, param.data, state['m'], state['v'])

    def update(self, lossfun=None, *args, **kwds):
        # Overwrites GradientMethod.update in order to get loss values
        if lossfun is None:
            raise RuntimeError('Eve.update requires lossfun to be specified')
        loss_var = lossfun(*args, **kwds)
        self.loss = float(loss_var.data)
        super(Eve, self).update(lossfun=lambda: loss_var)

    @property
    def lr(self):  # lr: learning rate
        fix1 = 1. - self.beta1 ** self.t
        fix2 = 1. - self.beta2 ** self.t
        # -> seems to be equal to \alpha (decay?)
        return self.alpha * math.sqrt(fix2) / fix1
