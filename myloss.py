from mxnet.gluon import loss as gloss
from mxnet.gluon.loss import Loss, _reshape_like, _apply_weighting


class L1Loss(Loss):
    r"""Calculates the mean absolute error between `label` and `pred`.

    .. math:: L = \sum_i \vert {label}_i - {pred}_i \vert.

    `label` and `pred` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.abs(label - pred)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss)


class L2Loss(Loss):
    r"""Calculates the mean squared error between `label` and `pred`.

    .. math:: L = \frac{1}{2} \sum_i \vert {label}_i - {pred}_i \vert^2.

    `label` and `pred` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.square(label - pred)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss)


class SigmoidBinaryCrossEntropyLoss(Loss):
    r"""The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)

    BCE loss is useful when training logistic regression. If `from_sigmoid`
    is False (default), this loss computes:

    .. math::

        prob = \frac{1}{1 + \exp(-{pred})}

        L = - \sum_i {label}_i * \log({prob}_i) * pos\_weight +
            (1 - {label}_i) * \log(1 - {prob}_i)

    If `from_sigmoid` is True, this loss computes:

    .. math::

        L = - \sum_i {label}_i * \log({pred}_i) * pos\_weight +
            (1 - {label}_i) * \log(1 - {pred}_i)

    A tensor `pos_weight > 1` decreases the false negative count, hence increasing
    the recall.
    Conversely setting `pos_weight < 1` decreases the false positive count and
    increases the precision.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    from_sigmoid : bool, default is `False`
        Whether the input is from the output of sigmoid. Set this to false will make
        the loss calculate sigmoid and BCE together, which is more numerically
        stable through log-sum-exp trick.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with values in range `[0, 1]`. Must have the
          same size as `pred`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).
        - **pos_weight**: a weighting tensor of positive examples. Must be a vector with length
          equal to the number of classes.For example, if pred has shape (64, 10),
          pos_weight should have shape (1, 10).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, pred, label, sample_weight=None, pos_weight=None):
        label = _reshape_like(F, label, pred)
        if not self._from_sigmoid:
            if pos_weight is None:
                # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
                loss = F.relu(pred) - pred * label + \
                    F.Activation(-F.abs(pred), act_type='softrelu')
            else:
                # We use the stable formula: x - x * z + (1 + z * pos_weight - z) * \
                #    (log(1 + exp(-abs(x))) + max(-x, 0))
                log_weight = 1 + F.broadcast_mul(pos_weight - 1, label)
                loss = pred - pred * label + log_weight * \
                       (F.Activation(-F.abs(pred), act_type='softrelu') + F.relu(-pred))
        else:
            eps = 1e-12
            if pos_weight is None:
                loss = -(F.log(pred + eps) * label
                         + F.log(1. - pred + eps) * (1. - label))
            else:
                loss = -(F.broadcast_mul(F.log(pred + eps) * label, pos_weight)
                         + F.log(1. - pred + eps) * (1. - label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss)


SigmoidBCELoss = SigmoidBinaryCrossEntropyLoss
