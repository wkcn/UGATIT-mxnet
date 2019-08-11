import numpy as np
import mxnet as mx
import time

ctx = mx.cpu()
mx.test_utils.set_default_context(ctx)


def test(func):
    T = 1000
    for _ in range(10):
        b = func(a)
    mx.nd.waitall()

    tic = time.time()
    for _ in range(T):
        b = func(a)
    mx.nd.waitall()
    print(time.time() - tic)
    return b

a = mx.random.normal(shape=(10, 10, 100, 100))


def test_avg_pool():
    func = lambda a : mx.nd.Pooling(a, kernel=(1, 1), pool_type='avg', global_pool=True)
    out1 = test(func)

    func = lambda a : mx.nd.mean(a, axis=(2, 3), keepdims=True)
    out2 = test(func)

    # Best on CPU
    func = lambda a : mx.nd.contrib.AdaptiveAvgPooling2D(a, output_size=(1, 1))
    out3 = test(func)

    np.testing.assert_allclose(out1.asnumpy(), out2.asnumpy(), atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(out1.asnumpy(), out3.asnumpy(), atol=1e-7, rtol=1e-5)


def test_max_pool():
    func = lambda a : mx.nd.Pooling(a, kernel=(1, 1), pool_type='max', global_pool=True)
    out1 = test(func)

    func = lambda a : mx.nd.max(a, axis=(2, 3), keepdims=True)
    out2 = test(func)

    np.testing.assert_allclose(out1.asnumpy(), out2.asnumpy(), atol=1e-7, rtol=1e-5)

print("global AVG Pool")
test_avg_pool()
print("global Max Pool")
test_max_pool()
