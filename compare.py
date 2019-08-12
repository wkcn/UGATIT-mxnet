import mxnet as mx
import torch
import numpy as np

import networks as mnn
import networks_th as tnn

name = 'ResnetBlock'
args = [3, True]
kwargs = {}
num_iter = 3

shape = (2, 3, 4, 4)
data = np.random.normal(size=shape).astype('float32')


def test_mx():
    class MyMXInit(mx.init.Initializer):
        def __init__(self):
            super(MyMXInit, self).__init__()
            self.value = 0.7
        def _init_weight(self, _, arr):
            arr[:] = self.value

    def force_init(params, init):
        for _, v in params.items():
            v.initialize(init, None, init, force_reinit=True)

    data_mx = mx.nd.array(data)
    data_mx.attach_grad()
    block = getattr(mnn, name)(*args, **kwargs)
    force_init(block.collect_params(), MyMXInit())
    with mx.autograd.record():
        out = block(data_mx)
        out.sum().backward()
    return out.asnumpy(), data_mx.grad.asnumpy()

def test_th():
    data_th = torch.tensor(data)
    data_th.requires_grad = True
    block = getattr(tnn, name)(*args, **kwargs)
    for v in block.parameters():
        torch.nn.init.constant_(v, 0.7)
    out = block(data_th)
    loss = out.sum()
    loss.backward()
    return out.detach().numpy(), data_th.grad.numpy()


out_mx, grad_mx = test_mx()
out_th, grad_th = test_th()
print(out_mx, out_th)
print('===========')
print(grad_mx, grad_th)
