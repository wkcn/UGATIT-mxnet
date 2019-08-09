import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn

EPSILON = 1e-08
POWER_ITERATION = 1

def normalize(x, dim, eps):
    norm = x.norm(axis=dim, keepdims=True)
    return x / mx.nd.maximum(norm, eps)

@mx.init.register
class SNUVInit(mx.init.Initializer):
    def __init__(self):
        super(SNUVInit, self).__init__()
    def _init_weight(self, name, arr):
        arr[:] = normalize(
            mx.random.normal(shape=arr.shape, dtype=arr.dtype),
            dim=0,
            eps=1e-12
        )

def _register_spectral_norm(name, cls):

    def __init__(self, *args, **kwargs):
        self._parent_cls = super(self.__class__, self)
        self._parent_cls.__init__(*args, **kwargs)
        self.iterations = POWER_ITERATION
        self.eps = 1e-12
        shape = self.weight.shape
        nw = np.prod(shape[1:]) if len(shape) > 1 else 1
        self.weight_u = self.params.get('weight_u', init=SNUVInit(), shape=(shape[0], ))
        self.weight_v = self.params.get('weight_v', init=SNUVInit(), shape=(nw,))

    def forward(self, x): #, weight, bias, extra_u):
        F = mx.nd
        weight = self.weight.data()
        weight_mat = weight.flatten()
        bias = self.bias.data() if self.bias is not None else None
        u = self.weight_u.data()
        v = self.weight_v.data()
        with autograd.pause():
            for _ in range(POWER_ITERATION):
                v[:] = normalize(mx.nd.dot(weight_mat.transpose(), u), dim=0, eps=self.eps)
                u[:] = normalize(mx.nd.dot(weight_mat, v), dim=0, eps=self.eps)
        sigma = mx.nd.dot(u, mx.nd.dot(weight_mat, v))
        weight = weight / sigma
        if bias is not None:
            return self._parent_cls.hybrid_forward(F, x, weight, bias)
        else:
            return self._parent_cls.hybrid_forward(F, x, weight)

    inst_dict = dict(
        __init__=__init__,
        # hybrid_forward=hybrid_forward
        forward=forward,
    )
    inst = type(name, (cls, ), inst_dict)
    globals()[name] = inst

_register_spectral_norm('SNConv2D', nn.Conv2D)
_register_spectral_norm('SNDense', nn.Dense)


if __name__ == '__main__':
    import mxnet as mx
    import numpy as np
    import torch
    from torch import nn as tnn
    channels = 2
    kernel_size = (2, 2)
    N, C, H, W = 1, 1, 3, 3
    conv_mx = SNConv2D(channels=channels, kernel_size=kernel_size, use_bias=False, in_channels=C)
    data_np = np.random.normal(size=(N, C, H, W)).astype('float32')
    weight_np = np.random.normal(size=(channels, C) + kernel_size).astype('float32')
    data_mx = mx.nd.array(data_np) 
    weight_mx = mx.nd.array(weight_np) 
    conv_mx.initialize()
    conv_mx.weight.data()[:] = weight_mx
    lr = 1e-2
    wd = 0.9
    trainer_mx = mx.gluon.Trainer(conv_mx.collect_params(), 'adam', dict(learning_rate=lr, beta1=0.5, beta2=0.999, wd=wd))
    target = 10
    iters = 4
    for _ in range(iters):
        with mx.autograd.record():
            out_mx = conv_mx(data_mx)
            sum_mx = out_mx.sum()
            loss_mx = (target - sum_mx).square()
            loss_mx.backward()
        trainer_mx.step(1)
        print(out_mx, "KS")#, conv_mx.weight.data(), "U", conv_mx.weight_u.data(), conv_mx.weight_v.data())

    print('-----------------------')

    conv_th = tnn.utils.spectral_norm(tnn.Conv2d(C, channels, kernel_size=kernel_size, bias=False)) 
    conv_th.weight_orig.data = torch.tensor(weight_np)
    data_th = torch.tensor(data_np)
    trainer_th = torch.optim.Adam(conv_th.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    for _ in range(iters):
        trainer_th.zero_grad()
        out_th = conv_th(data_th)
        sum_th = out_th.sum()
        loss_th = (target - sum_th) ** 2
        loss_th.backward()
        trainer_th.step()
        print(out_th, "KS")#, conv_th.weight.data, "U", conv_th.weight_u, conv_th.weight_v)
