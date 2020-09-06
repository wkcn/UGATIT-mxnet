import mxnet as mx
import torch
import numpy as np

import networks as mnn
import networks_th as tnn

import functools
import operator

mx.debug_spnorm = True
torch.debug_spnorm = True
PARAMS = {}

'''
[x] ResnetGenerator
    gradient may be wrong
[x] ResnetBlock
    gradient match after replacing instance norm
[x] ResnetAdaILNBlock
    gradient match
[x] adaILN
    only used in ResnetAdaILNBlock
[x] ILN
[x] Discriminator
    gradient match
'''

NO = slice(None, None, None)
NI = 1

'''
name = 'Discriminator'
args = [3, 3, 4]
'''

name = 'ILN'
args = [3]

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
UU = 1e10

shape = (2, 3, 64, 64)
data = np.random.normal(size=shape).astype('float32')
gamma = np.random.normal(size=shape[:2]).astype('float32')
beta = np.random.normal(size=shape[:2]).astype('float32')


alpha = 1e-7
def test_mx():
    class MyMXInit(mx.init.Initializer):
        def __init__(self):
            super(MyMXInit, self).__init__()
            self.value = alpha 
        def _init_weight(self, _, arr):
            shp = arr.shape
            if shp not in PARAMS:
                PARAMS[shp] = np.random.normal(size=shp).astype('float32')
                assert PARAMS[shp].shape == shp
            arr[:] = mx.nd.array(PARAMS[shp])
            # arr[:] = self.value * (mx.nd.arange(arr.size).reshape_like(arr) - arr.size/2.0)
            # arr[:] = (arr - arr.mean()) / (arr.max() - arr.min()) - 0.5

    def force_init(params, init):
        for _, v in params.items():
            v.initialize(init, None, init, force_reinit=True)

    data_mx = mx.nd.array(data)
    gamma_mx = mx.nd.array(gamma)
    beta_mx = mx.nd.array(beta)
    data_mx.attach_grad()
    inputs = [data_mx, gamma_mx, beta_mx]
    block = getattr(mnn, name)(*args, **kwargs)
    # block.collect_params().initialize()
    # force_init(block.collect_params('.*?_weight'), MyMXInit())
    force_init(block.collect_params(), MyMXInit())
    with mx.autograd.record():
        out = block(*inputs[:NI])[NO]
        (out.sum()*UU).backward()
    return out.asnumpy(), data_mx.grad.asnumpy()

def test_th():
    data_th = torch.tensor(data)
    data_th.requires_grad = True
    gamma_th = torch.tensor(gamma)
    beta_th = torch.tensor(beta)
    inputs = [data_th, gamma_th, beta_th]
    block = getattr(tnn, name)(*args, **kwargs)
    for k, v in block.named_parameters():
        if True or 'weight' in k:
            with torch.no_grad():
                shp = tuple(v.shape)
                v[:] = torch.tensor(PARAMS[shp])
                # v[:] = ((torch.arange(v.numel()).float().reshape(v.shape) - v.numel()/2.0)) * alpha 
                #v[:] = (v - v.mean()) / (v.max() - v.min()) - 0.5
    print (list(block.parameters()))
    out = block(*inputs[:NI])[NO]
    loss = out.sum() * UU
    loss.backward()
    return out.detach().numpy(), data_th.grad.numpy()

atol = 1e-4
rtol = 1e-4

out_mx, grad_mx = test_mx()
out_th, grad_th = test_th()


print(out_mx.max(), out_mx.min(), out_mx.mean(), grad_mx.mean())
print(out_th.max(), out_th.min(), out_th.mean(), grad_th.mean())
print('----')
print(grad_th.mean() / grad_mx.mean())
grad_diff = grad_mx - grad_th
print("GRAD", np.abs(grad_diff).mean(), np.abs(grad_diff).max())

np.testing.assert_allclose(out_mx, out_th, atol=atol, rtol=rtol)
print('===========')
np.testing.assert_allclose(grad_mx, grad_th, atol=atol, rtol=rtol)
print("PASS")
