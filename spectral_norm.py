import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn


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


class SpectralNormWeight(mx.operator.CustomOp):
    def __init__(self, num_iter, eps):
        super(SpectralNormWeight, self).__init__()
        self.num_iter = num_iter
        self.eps = eps

    def forward(self, is_train, req, in_data, out_data, aux):
        weight = in_data[0]
        state_u, state_v = aux
        weight_mat = weight.flatten()
        with autograd.pause():
            for _ in range(self.num_iter):
                state_v[:] = normalize(
                    mx.nd.dot(weight_mat.transpose(), state_u), dim=0, eps=self.eps)
                state_u[:] = normalize(
                    mx.nd.dot(weight_mat, state_v), dim=0, eps=self.eps)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assigin(in_grad[0], req[0], 0)


@mx.operator.register('SpectralNormWeight')
class SpectralNormWeightProp(mx.operator.CustomOpProp):
    def __init__(self, num_iter=1, eps=1e-12):
        super(SpectralNormWeightProp, self).__init__(need_top_grad=True)
        self.num_iter = int(num_iter)
        self.eps = float(eps)

    def list_arguments(self):
        return ['weight']

    def list_outputs(self):
        return ['dummy']

    def infer_shape(self, in_shape):
        assert len(in_shape) == 1, len(in_shape)
        shape = in_shape[0]
        nw = np.prod(shape[1:]) if len(shape) > 1 else 1
        aux_shape = [(shape[0], ), (nw, )]
        return in_shape, [(1,)], aux_shape

    def infer_type(self, in_type):
        dtype = in_type[0]
        return in_type, in_type, [dtype, dtype]

    def list_arguments(self):
        return ['weight']

    def list_auxiliary_states(self):
        return ['state_u', 'state_v']

    def create_operator(self, ctx, shapes, dtypes):
        return SpectralNormWeight(self.num_iter, self.eps)


def _register_spectral_norm(name, cls):
    def __init__(self, *args, **kwargs):
        self._parent_cls = super(self.__class__, self)
        self._parent_cls.__init__(*args, **kwargs)
        shape = self.weight.shape
        nw = np.prod(shape[1:]) if len(shape) > 1 else 1
        self.state_u = self.params.get(
            'state_u', grad_req='null', init=SNUVInit(), shape=(shape[0], ))
        self.state_v = self.params.get(
            'state_v', grad_req='null', init=SNUVInit(), shape=(nw,))

    def hybrid_forward(self, F, x, weight, bias=None, state_u=None, state_v=None):
        # update state_u and state_v
        F.Custom(weight, state_u, state_v,
                 op_type='SpectralNormWeight')
        weight_mat = weight.flatten()
        sigma = F.dot(state_u, F.dot(weight_mat, state_v))
        norm_weight = F.broadcast_div(weight, sigma)
        if bias is not None:
            return self._parent_cls.hybrid_forward(F, x, norm_weight, bias)
        else:
            return self._parent_cls.hybrid_forward(F, x, norm_weight)

    inst_dict = dict(
        __init__=__init__,
        hybrid_forward=hybrid_forward
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
    conv_mx = SNConv2D(channels=channels,
                       kernel_size=kernel_size, use_bias=False, in_channels=C)
    data_np = np.random.normal(size=(N, C, H, W)).astype('float32')
    weight_np = np.random.normal(
        size=(channels, C) + kernel_size).astype('float32')
    data_mx = mx.nd.array(data_np)
    weight_mx = mx.nd.array(weight_np)
    conv_mx.initialize()
    conv_mx.weight.data()[:] = weight_mx
    lr = 1e-2
    wd = 0.9
    trainer_mx = mx.gluon.Trainer(conv_mx.collect_params(), 'adam', dict(
        learning_rate=lr, beta1=0.5, beta2=0.999, wd=wd))
    target = 10
    iters = 5
    for _ in range(iters):
        with mx.autograd.record():
            out_mx = conv_mx(data_mx)
            sum_mx = out_mx.sum()
            loss_mx = (target - sum_mx).square()
            loss_mx.backward()
        trainer_mx.step(1)
        # , conv_mx.weight.data(), "U", conv_mx.weight_u.data(), conv_mx.weight_v.data())
        print(out_mx, "KS")

    print('-----------------------')

    conv_th = tnn.utils.spectral_norm(tnn.Conv2d(
        C, channels, kernel_size=kernel_size, bias=False))
    conv_th.weight_orig.data = torch.tensor(weight_np)
    data_th = torch.tensor(data_np)
    trainer_th = torch.optim.Adam(
        conv_th.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    for _ in range(iters):
        trainer_th.zero_grad()
        out_th = conv_th(data_th)
        sum_th = out_th.sum()
        loss_th = (target - sum_th) ** 2
        loss_th.backward()
        trainer_th.step()
        # , conv_th.weight.data, "U", conv_th.weight_u, conv_th.weight_v)
        print(out_th, "KS")
