import numpy as np

from ndnn.node import Node


class Attention(Node):
    def __init__(self, inputs, hidden):
        super().__init__([inputs])
        self.inputs = inputs
        self.hidden = hidden

    def compute(self):
        dotted = np.einsum('bln,bn->bl', self.inputs.value, self.hidden.value)

        lmax = np.max(dotted, axis=-1, keepdims=True)
        ex = np.exp(dotted - lmax)
        # B, L
        self.softmax = ex / np.sum(ex, axis=-1, keepdims=True)

        # BL, BLN -> BN
        return np.einsum('bl,bln->bn', self.softmax, self.inputs.value)

    def updateGrad(self):
        smaxgrad = np.einsum('bn,bln->bl', self.grad, self.inputs.value)
        inputsgrad = np.einsum('bn,bl->bln', self.grad, self.softmax)

        self.inputs.grad += inputsgrad

        gvdot = np.matmul(smaxgrad[..., np.newaxis, :], self.softmax[..., np.newaxis]).squeeze(-1)
        # BL
        dotgrad = self.softmax * (smaxgrad - gvdot)

        self.hidden.grad += np.einsum('bl,bln->bn', dotgrad, self.inputs.value)
        self.inputs.grad += np.einsum('bl,bn->bln', dotgrad, self.hidden.value)
