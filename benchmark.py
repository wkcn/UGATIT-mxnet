import mxnet as mx
import torch
import numpy as np
torch.backends.cudnn.benchmark = True

import networks as mnn
import networks_th as tnn

import functools
import operator
import time

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

NO = 0 #slice(None, None, None)
NI = 1

name = None
args = None

PRE_TIMES = 20
FORWARD_TIMES = 100
BACKWARD_TIMES = 100

USE_GPU = 1

def change_config(flag):
    global name, args, NI, NO
    if flag == 0:
        name = 'Discriminator'
        args = [3, 3, 4]
        NI = 1
    elif flag == 1:
        name = 'ILN'
        args = [3]
        NI = 1
    elif flag == 2:
        name = 'ResnetAdaILNBlock'
        args = [3, True]
        NI = 3
    elif flag == 3:
        name = 'adaILN'
        args = [3]
        NI = 3
    elif flag == 4:
        name = 'ResnetGenerator'
        args = [3, 5, 2, 3, 64, False]
        NI = 1
    elif flag == 5:
        name = 'ResnetBlock'
        args = [3, 5]
        NI = 1

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
    block = getattr(mnn, name)(*args, **kwargs)
    # block.collect_params().initialize()
    # force_init(block.collect_params('.*?_weight'), MyMXInit())
    force_init(block.collect_params(), MyMXInit())
    if USE_GPU:
        ctx = mx.gpu()
        data_mx = data_mx.as_in_context(ctx)
        gamma_mx = gamma_mx.as_in_context(ctx)
        beta_mx = beta_mx.as_in_context(ctx)
        block.collect_params().reset_ctx(ctx)
    inputs = [data_mx, gamma_mx, beta_mx]
    block.hybridize()
    for _ in range(PRE_TIMES):
        out = block(*inputs[:NI])[NO]
        with mx.autograd.record():
            out = block(*inputs[:NI])[NO]
            (out.sum()*UU).backward()
    mx.nd.waitall()
    fwd_tic = time.time()
    for _ in range(FORWARD_TIMES):
        out = block(*inputs[:NI])[NO]
        mx.nd.waitall()
    fwd_time = time.time() - fwd_tic
    back_tic = time.time()
    for _ in range(BACKWARD_TIMES):
        with mx.autograd.record():
            out = block(*inputs[:NI])[NO]
            (out.sum()*UU).backward()
        mx.nd.waitall()
    back_time = time.time() - back_tic
    return fwd_time, back_time, fwd_time+back_time

def test_th():
    data_th = torch.tensor(data)
    data_th.requires_grad = True
    gamma_th = torch.tensor(gamma)
    beta_th = torch.tensor(beta)
    block = getattr(tnn, name)(*args, **kwargs)
    if USE_GPU:
        data_th = data_th.cuda()
        gamma_th = gamma_th.cuda()
        beta_th = beta_th.cuda()
        block = block.cuda()
    inputs = [data_th, gamma_th, beta_th]
    for k, v in block.named_parameters():
        if True or 'weight' in k:
            with torch.no_grad():
                shp = tuple(v.shape)
                v[:] = torch.tensor(PARAMS[shp])
                if USE_GPU:
                    v[:] = v.cuda()
                # v[:] = ((torch.arange(v.numel()).float().reshape(v.shape) - v.numel()/2.0)) * alpha 
                #v[:] = (v - v.mean()) / (v.max() - v.min()) - 0.5
    for _ in range(PRE_TIMES):
        out = block(*inputs[:NI])[NO]
        loss = out.sum() * UU
        loss.backward()
    fwd_tic = time.time()
    data_th = torch.tensor(data)
    for _ in range(FORWARD_TIMES):
        out = block(*inputs[:NI])[NO]
    fwd_time = time.time() - fwd_tic
    data_th.requires_grad = True 
    back_tic = time.time()
    for _ in range(BACKWARD_TIMES):
        out = block(*inputs[:NI])[NO]
        loss = out.sum() * UU
        loss.backward()
    back_time = time.time() - back_tic
    return fwd_time, back_time, fwd_time+back_time

atol = 1e-4
rtol = 1e-4


start_flag = 0
for flag in range(start_flag, 6):
    change_config(flag)
    print('============={}: {}============='.format(flag, name))
    for _ in range(3):
        print("MX", test_mx())
        print("TH", test_th())
