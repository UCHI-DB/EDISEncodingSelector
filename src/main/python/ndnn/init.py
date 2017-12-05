import numpy as np


class Xavier(object):
    def __init__(self):
        pass

    def apply(self, shape):
        sq = np.sqrt(3.0 / np.prod(shape[:-1]))
        return np.random.uniform(-sq, sq, shape)


class Zero(object):
    def __init__(self):
        pass

    def apply(self, shape):
        return np.zeros(shape)
