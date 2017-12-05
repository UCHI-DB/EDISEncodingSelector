import numpy as np

etaDefault = 0.5
etaDecay = 1

gradClip = -1

epsilon = np.float64(1e-8)

momentumKey = "momentum"
mAlpha = 0.9

adagradKey = "adagrad"

rmspropKey = "rmsprop"
rmspropBeta = 0.9

adammeanKey = "adammean"
adamvarKey = "adamvar"
adamAlpha = 0.9
adamBeta = 0.999


class UpdatePolicy(object):
    def __init__(self):
        pass

    def update(self, param):
        self.clip_grad(param)
        self.inner_update(param)

    def weight_decay(self):
        self.eta *= self.decay

    def clip_grad(self, param):
        if self.grad_clip > 0:
            norm = np.linalg.norm(param.grad)
            if norm >= self.grad_clip:
                param.grad *= self.grad_clip / norm


class SGD(UpdatePolicy):
    def __init__(self, eta=etaDefault, decay=etaDecay, gc=gradClip):
        super().__init__()
        self.eta = eta
        self.decay = decay
        self.grad_clip = gc

    def inner_update(self, param):
        param.value -= param.grad * self.eta


class Momentum(UpdatePolicy):
    def __init__(self, eta=etaDefault, decay=etaDecay, alpha=mAlpha, gc=gradClip):
        super().__init__()
        self.eta = eta
        self.decay = decay
        self.alpha = alpha
        self.grad_clip = gc

    def inner_update(self, param):
        if momentumKey in param.env:
            momentum = param.env[momentumKey]
        else:
            momentum = np.float64(0)

        momentum = momentum * self.alpha + param.grad
        param.value -= momentum * self.eta
        param.env[momentumKey] = momentum


class AdaGrad(UpdatePolicy):
    def __init__(self, e=etaDefault, d=etaDecay, gc=gradClip):
        super().__init__()
        self.eta = e
        self.decay = d
        self.grad_clip = gc

    def inner_update(self, param):
        if adagradKey in param.env:
            adagrad = param.env[adagradKey]
        else:
            adagrad = np.float64(0)

        adagrad += param.grad * param.grad
        param.value -= self.eta * self.grad / np.sqrt(adagrad + epsilon)
        param.env[adagradKey] = adagrad


class RMSProp(UpdatePolicy):
    def __init__(self, eta=etaDefault, decay=etaDecay, beta=rmspropBeta, gc=gradClip):
        super().__init__()
        self.eta = eta
        self.decay = decay
        self.beta = beta
        self.grad_clip = gc

    def inner_update(self, param):
        gradsqr = np.power(param.grad, 2)
        if rmspropKey in param.env:
            oldrms = param.env[rmspropKey]
        else:
            oldrms = np.float64(0)
        rms = oldrms * self.beta + gradsqr * (1 - self.beta)
        param.value -= param.grad * self.eta / np.sqrt(rms + epsilon)
        param.env[rmspropKey] = rms


class Adam(UpdatePolicy):
    def __init__(self, eta=etaDefault, decay=etaDecay, alpha=adamAlpha, beta=adamBeta, gc=gradClip):
        super().__init__()
        self.eta = eta
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.grad_clip = gc

    def inner_update(self, param):
        if adammeanKey in param.env:
            oldmomen = param.env[adammeanKey]
        else:
            oldmomen = 0
        momentum = oldmomen * self.alpha + param.grad * (1 - self.alpha)

        gradsqr = np.power(param.grad, 2)
        if adamvarKey in param.env:
            oldrms = param.env[adamvarKey]
        else:
            oldrms = 0
        rms = oldrms * self.beta + gradsqr * (1 - self.beta)
        param.value -= momentum * self.eta / np.sqrt(rms + epsilon)

        param.env[adammeanKey] = momentum
        param.env[adamvarKey] = rms


'''
Dense-Sparse-Dense Training
See https://arxiv.org/pdf/1607.04381.pdf

The training process contains 3 phases
* phase 1: normal training
* phase 2: apply a watermark to weight and update only those above watermarks
* phase 3: remove the watermark and train normally
'''
maskKey = "weight.mask"
threshold = 0.001


class DSD(UpdatePolicy):
    def __init__(self, childpolicy, phase1, phase2):
        super().__init__()
        self.child = childpolicy
        self.phase1 = phase1
        self.phase2 = phase2
        self.current_epoch = 0

    def inner_update(self, param):
        self.child.inner_update(param)
        if self.phase1 <= self.current_epoch < self.phase2:
            # Apply mask
            if maskKey not in param.env:
                mask = np.greater(np.abs(param.value), threshold)
                param.env[maskKey] = mask
            else:
                mask = param.env[maskKey]
            param.value *= mask

    def clip_grad(self, param):
        self.child.clip_grad(param)

    def weight_decay(self):
        self.child.weight_decay()
        self.current_epoch += 1
