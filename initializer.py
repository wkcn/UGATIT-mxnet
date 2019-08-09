import mxnet as mx
import math


class KaimingUniform(mx.initializer.Xavier):
    """Initialize the weight according to a MSRA paper.

    This initializer implements *Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification*, available at
    https://arxiv.org/abs/1502.01852.

    This initializer is proposed for initialization related to ReLu activation,
    it maked some changes on top of Xavier method.

    Parameters
    ----------
    factor_type: str, optional
        Can be ``'avg'``, ``'in'``, or ``'out'``.

    slope: float, optional
        initial slope of any PReLU (or similar) nonlinearities.
    """

    def __init__(self, factor_type="in", slope=math.sqrt(5)):
        magnitude = 6. / (1 + slope ** 2)
        super(KaimingUniform, self).__init__("uniform", factor_type, magnitude)
        self._kwargs = {'factor_type': factor_type, 'slope': slope}


class BiasInitializer(mx.initializer.Initializer):
    def __init__(self, params):
        super(BiasInitializer, self).__init__()
        self.params = params

    def _init_weight(self, name, arr):
        shape = self.params[name.replace('_bias', '_weight')].shape
        hw_scale = 1.
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        bound = 1. / math.sqrt(fan_in)
        mx.random.uniform(-bound, bound, out=arr)
