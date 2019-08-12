import mxnet as mx
import torch
import numpy as np

import networks as mnn
import networks_th as tnn

import functools
import operator

'''
[x] ResnetGenerator
    gradient may be wrong
[x] ResnetBlock
    replace instance norm
[x] ResnetAdaILNBlock
    good gradient
[x] adaILN
[x] ILN
[ ] Discriminator
'''

NI = 1

name = 'Discriminator'
args = [3, 3, 4]

'''
name = 'ILN'
args = [3]
'''

'''
name = 'ResnetAdaILNBlock'
args = [3, True]
NI = 3
'''

'''
name = 'adaILN'
args = [3]
NI = 3
'''

'''
name = 'ResnetGenerator'
args = [3, 5, 2, 3, 64, False]
'''

'''
name = 'ResnetBlock'
args = [3, 5]
'''

'''
name = 'TestInstanceNorm'
args = [3]
'''

kwargs = {}
num_iter = 3
UU = 100

shape = (2, 3, 64, 64)
data = np.random.normal(size=shape).astype('float32')
gamma = np.random.normal(size=shape[:2]).astype('float32')
beta = np.random.normal(size=shape[:2]).astype('float32')


alpha = 1e-3
def test_mx():
    class MyMXInit(mx.init.Initializer):
        def __init__(self):
            super(MyMXInit, self).__init__()
            self.value = alpha 
        def _init_weight(self, _, arr):
            arr[:] = self.value * (mx.nd.arange(arr.size).reshape_like(arr) - arr.size/2.0)

    def force_init(params, init):
        for _, v in params.items():
            v.initialize(init, None, init, force_reinit=True)

    data_mx = mx.nd.array(data)
    gamma_mx = mx.nd.array(gamma)
    beta_mx = mx.nd.array(beta)
    data_mx.attach_grad()
    inputs = [data_mx, gamma_mx, beta_mx]
    block = getattr(mnn, name)(*args, **kwargs)
    force_init(block.collect_params(), MyMXInit())
    with mx.autograd.record():
        out = block(*inputs[:NI])
        (out.sum()*UU).backward()
    return out.asnumpy(), data_mx.grad.asnumpy()

def test_th():
    data_th = torch.tensor(data)
    data_th.requires_grad = True
    gamma_th = torch.tensor(gamma)
    beta_th = torch.tensor(beta)
    inputs = [data_th, gamma_th, beta_th]
    block = getattr(tnn, name)(*args, **kwargs)
    for v in block.parameters():
        with torch.no_grad():
            v[:] = ((torch.arange(v.numel()).float().reshape(v.shape) - v.numel()/2.0)) * alpha 
    out = block(*inputs[:NI])
    loss = out.sum() * UU
    loss.backward()
    return out.detach().numpy(), data_th.grad.numpy()

atol = 1e-4
rtol = 1e-4

out_mx, grad_mx = test_mx()
out_th, grad_th = test_th()


print(out_mx.max(), out_mx.min(), out_mx.mean(), grad_mx.mean())
print(out_th.max(), out_th.min(), out_th.mean(), grad_th.mean())
grad_diff = grad_mx - grad_th
print("GRAD", np.abs(grad_diff).mean(), np.abs(grad_diff).max())

np.testing.assert_allclose(out_mx, out_th, atol=atol, rtol=rtol)
print('===========')
#np.testing.assert_allclose(grad_mx, grad_th, atol=atol, rtol=rtol)
print("PASS")
