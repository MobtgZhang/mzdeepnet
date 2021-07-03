import numpy as np
from .layers import Layer
from mzdeepnet.base import PhaseMixin

class Dropout(Layer, PhaseMixin):
    def __init__(self, dropout=0.5):
        self.dropout = dropout
        self._tmp_mask = None

    def fprop(self, x):
        if self.dropout > 0.0:
            if self.phase == 'train':
                self._tmp_mask = self.dropout < np.random.uniform(size=x.shape)
                y = x * self._tmp_mask
            elif self.phase == 'test':
                y = x * (1.0 - self.dropout)
            else:
                raise ValueError('Invalid phase: %s' % self.phase)
        return y

    def bprop(self, y_grad):
        if self.dropout > 0.0:
            return y_grad * self._tmp_mask
        else:
            return y_grad

    def y_shape(self, x_shape):
        return x_shape
