import numpy as np

from ndnn.loss import Loss
from ndnn.loss import clip
from ndnn.node import Node


class LogLoss(Loss):
    def __init__(self):
        super().__init__()

    '''
    Actual is of shape [B, L, M]
    Expect is of shape [B, L]
    Should return an gradient of shape [B, L, M]    
    '''

    def loss(self, actual, expect, fortest):
        # The average loss is averaged to each slice
        all_batch_size = np.product(expect.shape)

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

        # Accuracy for classification is the number of corrected predicted items
        predict = np.argmax(actual, axis=-1)
        self.acc = np.equal(predict, expect).sum()

        # Record Error Detail
        if 'errorStat' in self.__dict__:
            pflat = predict.reshape(-1)
            eflat = expect.reshape(-1)
            for i in range(pflat.shape[0]):
                self.errorStat.add(int(pflat[i]), int(eflat[i]))

        return -np.log(clipval).mean()


class HingeLossOutput(Node):
    def __init__(self, actual, embed, neg):
        super().__init__([actual])
        self.embed = embed
        self.neg_sample = neg
        self.actual = actual

    def compute(self):
        return [self.actual.value, self.embed.value, self.neg_sample.value]

    def updateGrad(self):
        self.actual.grad += self.grad[0]
        self.embed.grad += self.grad[1]


class HingeLoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, actual, expect, fortest):
        # B, L, H
        ht = actual[0]
        b, l, h = ht.shape

        all_batch_size = b * l

        htflat = ht.reshape([b * l, -1])
        # N, H
        embed = actual[1]
        n = embed.shape[0]
        # R,
        neg_sample = actual[2]
        r = neg_sample.shape[0]

        # R, H
        neg_embed = embed[np.int32(neg_sample), :]
        # B * L, H
        expect_embed = embed[np.int32(expect.reshape([-1])), :].reshape([b * l, -1])
        # Compute Loss max(1-y_t^Th_t + y'^Th_t, 0)
        # B * L, 1
        ytht = np.expand_dims(np.einsum('bh,bh->b', htflat, expect_embed), 1)
        # B * L, R
        ypht = np.matmul(htflat, neg_embed.T)
        # B * L, R
        loss = np.maximum(0, 1 - ytht + ypht)

        mask = loss > 0

        # Gradient
        if not fortest:
            # h_t = \sum_y' (y' - y_t)
            predict_grad = (np.matmul(mask, neg_embed) -
                            expect_embed * mask.sum(axis=1, keepdims=True)) \
                .reshape([b, l, h])
            # B * L, H
            ytgrad = - mask.sum(axis=1, keepdims=True) * htflat
            yt1 = np.zeros([n, b * l])
            yt1[expect.reshape(-1), np.arange(b * l)] = 1
            ytembedgrad = np.matmul(yt1, ytgrad)
            # R, H
            ypgrad = np.matmul(mask.T, htflat)
            yp1 = np.zeros([n, r])
            yp1[neg_sample, np.arange(r)] = 1
            ypembedgrad = np.matmul(yp1, ypgrad)
            self.grad = [predict_grad, (ytembedgrad + ypembedgrad)]

        # Accuracy
        # B * L, N
        predict = np.argmax(np.matmul(htflat, embed.T), 1)

        # Record Error Detail
        if 'errorStat' in self.__dict__:
            pflat = predict.reshape(-1)
            eflat = expect.reshape(-1)
            for i in range(pflat.shape[0]):
                self.errorStat.add(int(pflat[i]), int(eflat[i]))

        self.acc = (predict == expect.reshape(-1)).sum()

        return loss.sum()
