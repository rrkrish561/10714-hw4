from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        reduced_shape = list(Z.shape)
        reduced_shape[-1] = 1

        Z_max = array_api.max(Z, 1, keepdims=True)
        Z_max = array_api.reshape(Z_max, reduced_shape)
        Z_max = array_api.broadcast_to(Z_max, Z.shape)
        Z_sum = array_api.sum(array_api.exp(Z - Z_max), 1, keepdims=True)
        Z_sum = array_api.reshape(Z_sum, reduced_shape)
        Z_sum = array_api.broadcast_to(Z_sum, Z.shape)
        log_sum = array_api.log(Z_sum)
        log_sum_exp = Z_max + log_sum

        return Z - log_sum_exp

    def gradient(self, out_grad, node):
        sum_out_grad = broadcast_to(reshape(summation(out_grad, 1), (node.shape[0], 1)), node.shape)
        return out_grad - exp(node) * sum_out_grad


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        reduced_shape = list(Z.shape)
        if self.axes is None:
            reduced_shape[-1] = 1
        elif isinstance(self.axes, Number):
            reduced_shape[self.axes] = 1
        else:
            for x in self.axes:
                reduced_shape[x] = 1

        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_max = array_api.reshape(Z_max, reduced_shape)
        Z_max = array_api.broadcast_to(Z_max, Z.shape)
        Z_sum = array_api.sum(array_api.exp(Z - Z_max), axis=self.axes, keepdims=True)
        Z_sum = array_api.reshape(Z_sum, reduced_shape)
        Z_sum = array_api.broadcast_to(Z_sum, Z.shape)
        log_sum = array_api.log(Z_sum)

        # Count number of elements that we are averaging over
        count = 0
        if self.axes is None:
            count = Z.shape[-1]
        elif isinstance(self.axes, Number):
            count = Z.shape[self.axes]
        else:
            for x in self.axes:
                count += Z.shape[x] 

        return array_api.sum(Z_max + log_sum, axis=self.axes) / count

    def gradient(self, out_grad, node):
        inp_shape = node.inputs[0].shape

        new_shape = list(inp_shape)
        if self.axes is None:
            self.axes = tuple(range(len(new_shape)))
        elif isinstance(self.axes, Number):
            self.axes = (self.axes,)
        for x in self.axes:
            new_shape[x] = 1

        return broadcast_to(reshape(out_grad, new_shape), inp_shape) * exp(node.inputs[0] - broadcast_to(reshape(node, new_shape), node.inputs[0].shape))


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

