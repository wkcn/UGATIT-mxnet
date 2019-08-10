import mxnet as mx
import numpy as np
from mxnet import autograd
from mxnet.gluon import nn


def _register_record_hook(cls):
    def _hybrid_forward(self, F, x, weight, bias=None):
        self._weight = weight
        self._bias = bias
        return self._old_hybrid_forward(F, x, weight, bias)
    cls._old_hybrid_forward = cls.hybrid_forward
    cls.hybrid_forward = _hybrid_forward


_register_record_hook(nn.Dense)
_register_record_hook(nn.Conv2D)


def _mx_sym_getitem(self, index):
    assert isinstance(index, int)
    return mx.sym.take(self, mx.sym.full(val=index, shape=(1,)), 0)

mx.sym.Symbol.__getitem__ = _mx_sym_getitem
