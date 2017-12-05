import numpy as np


class Loss(object):
    def __init__(self):
        self.acc = None
        self.grad = None

    def loss(self, actual, expect, fortest):
        pass

    def grad(self):
        return self.grad

    def accuracy(self):
        return self.acc


class TrivialLoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, actual, expect, fortest):
        n = actual.shape[1]
        if not fortest:
            self.grad = np.ones_like(actual) / n
        self.acc = (actual == expect).sum()
        return actual.sum()


class SquareLoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, actual, expect, fortest):
        b = actual.shape()[0]

        if not fortest:
            # Compute Gradient
            self.grad = (actual - expect) * (1. / b)
        # Loss
        return np.power(actual - expect, 2).sum(axis=1).mean() / 2


clip = 1e-12


class LogLoss(Loss):
    def __init__(self):
        super().__init__()

    '''
    Actual shape is [B,1]
    Expect shape is [B,1]
    loss = -log(actual)*expect - log(1-actual)(1-expect)
    '''

    def loss(self, actual, expect, fortest):
        batch_size = expect.shape[0]
        clipped = np.maximum(actual, clip)

        nexpect = 1 - expect
        nclipped = 1 - clipped

        loss = -np.log(clipped * expect + nclipped * nexpect).mean()
        if not fortest:
            self.grad = (-expect / clipped + nexpect / nclipped) / batch_size

        predict = (actual >= 0.5)
        self.acc = np.equal(predict, expect).sum()

        return loss


class SoftMaxLoss(Loss):
    def __init__(self):
        super().__init__()

    '''
    Actual is of shape [A, B, ..., M]
    Expect is of shape [A, B, ..., 1]
    Should return an gradient of shape [A, B,...,M]    
    '''

    def loss(self, actual, expect, fortest):
        # The average loss is averaged to each slice
        all_batch_size = np.product(expect.shape)
        #        all_batch_size = expect.shape[1]

        xflat = actual.reshape(-1)
        iflat = expect.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        idx = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        fetch = xflat[idx].reshape(expect.shape)
        clipval = np.maximum(fetch, clip)

        if not fortest:
            # Compute Gradient
            slgrad = -np.ones_like(expect) / (clipval * all_batch_size)
            self.grad = np.zeros_like(actual)
            self.grad.reshape(-1)[idx] = slgrad

        # Accuracy for classification

        predict = np.argmax(actual, axis=-1)
        self.acc = np.equal(predict, expect).sum()

        return -np.log(clipval).mean()
