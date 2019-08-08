import mxnet as mx
import numpy as np
from mxnet.gluon import nn

def var(x, dim, keepdims=False, unbiased=True):
    s = (x - x.mean(dim, keepdims=True)).square().sum(dim, keepdims=keepdims)
    n = x.shape[dim]
    if unbiased:
        s /= n - 1
    else:
        s /= n
    return s


class ResnetGenerator(nn.HybridBlock):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [nn.ReflectionPad2D(3),
                      nn.Conv2D(ngf, kernel_size=7, strides=1, padding=0, use_bias=False),
                      nn.InstanceNorm(),
                      nn.Activation('relu')]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [nn.ReflectionPad2D(1),
                          nn.Conv2D(ngf * mult * 2, kernel_size=3, strides=2, padding=0, use_bias=False),
                          nn.InstanceNorm(),
                          nn.Activation('relu')]

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = nn.Dense(1, use_bias=False)
        self.gmp_fc = nn.Dense(1, use_bias=False)
        self.conv1x1 = nn.Conv2D(ngf * mult, kernel_size=1, strides=1, use_bias=True)
        self.relu = nn.Activation('relu')

        # Gamma, Beta block
        if self.light:
            FC = [nn.Dense(ngf * mult, use_bias=False),
                  nn.Activation('relu'),
                  nn.Dense(ngf * mult, use_bias=False),
                  nn.Activation('relu')]
        else:
            FC = [nn.Dense(ngf * mult, use_bias=False),
                  nn.Activation('relu'),
                  nn.Dense(ngf * mult, use_bias=False),
                  nn.Activation('relu')]
        self.gamma = nn.Dense(ngf * mult, use_bias=False)
        self.beta = nn.Dense(ngf * mult, use_bias=False)

        # Up-Sampling Bottleneck
        self.UpBlock1s = nn.HybridSequential()
        for i in range(n_blocks):
            self.UpBlock1s.add(ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')),
                         nn.ReflectionPad2D(1),
                         nn.Conv2D(int(ngf * mult / 2), kernel_size=3, strides=1, padding=0, use_bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.Activation('relu')]

        UpBlock2 += [nn.ReflectionPad2D(3),
                     nn.Conv2D(output_nc, kernel_size=7, strides=1, padding=0, use_bias=False),
                     nn.Activation('tanh')]

        self.DownBlock = nn.HybridSequential()
        self.DownBlock.add(*DownBlock)
        self.FC = nn.HybridSequential()
        self.FC.add(*FC)
        self.UpBlock2 = nn.HybridSequential()
        self.UpBlock2.add(*UpBlock2)

    def hybrid_forward(self, F, input):
        x = self.DownBlock(input)

        gap = F.contrib.AdaptiveAvgPooling2D(x, (1, 1))
        gap_logit = self.gap_fc(gap.reshape((x.shape[0], -1)))
        gap_weight = self.gap_fc.weight.data()
        gap = x * gap_weight.reshape((0, 0, 1, 1))

        gmp = F.contrib.AdaptiveMaxPooling2D(x, (1, 1))
        gmp_logit = self.gmp_fc(gmp.reshape((x.shape[0], -1)))
        gmp_weight = self.gmp_fc.weight.data()
        gmp = x * gmp_weight.reshape((0, 0, 1, 1))

        cam_logit = F.concat(*[gap_logit, gmp_logit], dim=1)
        x = F.concat(*[gap, gmp], dim=1)
        x = self.relu(self.conv1x1(x))

        heatmap = F.sum(x, axis=1, keepdims=True)

        if self.light:
            x_ = F.contrib.AdaptiveAvgPooling2D(x, (1, 1))
            x_ = self.FC(x_.reshape((x_.shape[0], -1)))
        else:
            x_ = self.FC(x.reshape((x.shape[0], -1)))
        gamma, beta = self.gamma(x_), self.beta(x_)


        for block in self.UpBlock1s:
            x = block(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(nn.HybridBlock):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2D(1),
                       nn.Conv2D(dim, kernel_size=3, strides=1, padding=0, use_bias=use_bias),
                       nn.InstanceNorm(),
                       nn.Activation('relu')]

        conv_block += [nn.ReflectionPad2D(1),
                       nn.Conv2D(dim, kernel_size=3, strides=1, padding=0, use_bias=use_bias),
                       nn.InstanceNorm()]

        self.conv_block = nn.HybridSequential()
        self.conv_block.add(*conv_block)

    def hybrid_forward(self, F, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.HybridBlock):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2D(1)
        self.conv1 = nn.Conv2D(dim, kernel_size=3, strides=1, padding=0, use_bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.Activation('relu')

        self.pad2 = nn.ReflectionPad2D(1)
        self.conv2 = nn.Conv2D(dim, kernel_size=3, strides=1, padding=0, use_bias=use_bias)
        self.norm2 = adaILN(dim)

    def hybrid_forward(self, F, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out


class adaILN(nn.HybridBlock):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = self.params.get('rho', shape=(1, num_features, 1, 1), init=mx.init.Constant(0.9))

    def hybrid_forward(self, F, input, gamma, beta, rho):
        in_mean, in_var = F.mean(input, (2, 3), keepdims=True), var(input, (2, 3), keepdims=True)
        out_in = (input - in_mean) / F.sqrt(in_var + self.eps)
        ln_mean, ln_var = F.mean(input, (1, 2, 3), keepdims=True), var(input, (1, 2, 3), keepdims=True)
        out_ln = (input - ln_mean) / F.sqrt(ln_var + self.eps)
        out = rho * out_in + (1 - rho) * out_ln
        out = out * gamma.reshape((0, 0, 1, 1)) + beta.reshape((0, 0, 1, 1))

        return out


class ILN(nn.HybridBlock):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = self.params.get('rho', shape=(1, num_features, 1, 1), init=mx.init.Constant(0.0))
        self.gamma = self.params.get('gamma', shape=(1, num_features, 1, 1), init=mx.init.Constant(1.0))
        self.beta = self.params.get('beta', shape=(1, num_features, 1, 1), init=mx.init.Constant(0.0))

    def hybrid_forward(self, F, input, rho, gamma, beta):
        in_mean, in_var = F.mean(input, (2, 3), keepdims=True), var(input, (2, 3), keepdims=True)
        out_in = (input - in_mean) / F.sqrt(in_var + self.eps)
        ln_mean, ln_var = F.mean(input, (1, 2, 3), keepdims=True), var(input, (1, 2, 3), keepdims=True)
        out_ln = (input - ln_mean) / F.sqrt(ln_var + self.eps)
        out = rho * out_in + (1 - rho) * out_ln
        out = out * gamma + beta

        return out


EPSILON = 1e-08
POWER_ITERATION = 1

def _spectral_norm(w, u, iterations):
    """ spectral normalization """
    w_mat = mx.nd.reshape(w, [w.shape[0], -1])

    _u = u
    _v = None

    for _ in range(iterations):
        _v = mx.nd.L2Normalization(nd.dot(_u, w_mat))
        _u = mx.nd.L2Normalization(nd.dot(_v, w_mat.T))

    sigma = mx.nd.sum(mx.nd.dot(_u, w_mat) * _v)
    if sigma == 0.:
        sigma = EPSILON

    with mx.autograd.pause():
        u[:] = _u

    return w / sigma

def _register_spectral_norm(name, cls):

    def __init__(self, *args, **kwargs):
        self._parent_cls = super(self.__class__, self)
        self._parent_cls.__init__(*args, **kwargs)
        self.iterations = POWER_ITERATION
        self.extra_u = self.params.get('extra_u', init=mx.init.Normal(), shape=(1, self.weight.shape[0]))


    def hybrid_forward(self, F, x, weight, *args):
        extra_u = args.pop()
        weight = _spectral_norm(weight, extra_u, self.iterations)
        self._parent_cls.hybrid_forward(F, x, weight, *args)

    inst_dict = dict(
        __init__=__init__,
        hybrid_forward=hybrid_forward
    )
    inst = type(name, (cls, ), inst_dict) 
    globals()[name] = inst

_register_spectral_norm('SNConv2D', nn.Conv2D)
_register_spectral_norm('SNDense', nn.Dense)


class Discriminator(nn.HybridBlock):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2D(1),
                 SNConv2D(ndf, kernel_size=4, strides=2, padding=0, use_bias=True),
                 nn.LeakyReLU(0.2)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2D(1),
                      SNConv2D(ndf * mult * 2, kernel_size=4, strides=2, padding=0, use_bias=True),
                      nn.LeakyReLU(0.2)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2D(1),
                  SNConv2D(ndf * mult * 2, kernel_size=4, strides=1, padding=0, use_bias=True),
                  nn.LeakyReLU(0.2)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = SNDense(1, use_bias=False)
        self.gmp_fc = SNDense(1, use_bias=False)
        self.conv1x1 = nn.Conv2D(ndf * mult, kernel_size=1, strides=1, use_bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.pad = nn.ReflectionPad2D(1)
        self.conv = SNConv2D(1, kernel_size=4, strides=1, padding=0, use_bias=False)

        self.model = nn.HybridSequential()
        self.model.add(*model)

    def hybrid_forward(self, F, input):
        x = self.model(input)

        gap = F.contrib.AdaptiveAvgPooling2D(x, (1, 1))
        gap_logit = self.gap_fc(gap.reshape((x.shape[0], -1)))
        gap_weight = self.gap_fc.weight.data()
        gap = x * gap_weight.reshape((0, 0, 1, 1))

        gmp = F.contrib.AdaptiveMaxPooling2D(x, (1, 1))
        gmp_logit = self.gmp_fc(gmp.reshape((x.shape[0], -1)))
        gmp_weight = self.gmp_fc.weight.data()
        gmp = x * gmp_weight.reshape((0, 0, 1, 1))

        cam_logit = F.concat(*[gap_logit, gmp_logit], dim=1)
        x = F.concat(*[gap, gmp], dim=1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = F.sum(x, axis=1, keepdims=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data()
            w = w.clip(self.clip_min, self.clip_max)
            module.rho.data()[:] = w
