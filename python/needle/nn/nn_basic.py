"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(weight, dtype=dtype, device=device)

        if bias:
            bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose()
            self.bias = Parameter(bias, dtype=dtype, device=device)

    def forward(self, X: Tensor) -> Tensor:
        res = ops.matmul(X, self.weight)
        if self.bias is None:
            return res
        else:
            bias_shape = (1,) * (len(res.shape) - 1) + (self.bias.shape[-1],) 
            return res + ops.broadcast_to(ops.reshape(self.bias, bias_shape), res.shape)


class Flatten(Module):
    def forward(self, X):
        y_dim = 1
        for i in range(1, len(X.shape)):
            y_dim *= X.shape[i]
        return ops.reshape(X, (X.shape[0], y_dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
    
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        next = x
        for module in self.modules:
            next = module(next)
        return next


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        one_hot = init.one_hot(logits.shape[1], y, dtype=logits.dtype, device=logits.device)

        return ops.summation(ops.logsumexp(logits, 1) - ops.summation(logits * one_hot, 1)) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype), dtype=dtype, device=device)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype), dtype=dtype, device=device)

        self.running_mean = init.zeros(dim, dtype=dtype, device=device).detach()
        self.running_var = init.ones(dim, dtype=dtype, device=device).detach()

    def forward(self, x: Tensor) -> Tensor:
        if(self.training):
            E = ops.summation(x, 0) / x.shape[0]
            V = ops.summation((x - ops.broadcast_to(ops.reshape(E, (1, self.dim)), x.shape)) ** 2, 0) / x.shape[0]

            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * E).detach()
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * V).detach()

            E = ops.broadcast_to(ops.reshape(E, (1, self.dim)), x.shape)
            V = ops.broadcast_to(ops.reshape(V, (1, self.dim)), x.shape)
            V_eps = V + self.eps

            return ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) * (x - E) / ops.power_scalar(V_eps, 1/2) + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        else:
            broadcast_mean = ops.broadcast_to(ops.reshape(self.running_mean, (1, self.dim)), x.shape).data
            broadcast_var = ops.broadcast_to(ops.reshape(self.running_var, (1, self.dim)), x.shape).data
            return ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) * (x - broadcast_mean) / ops.power_scalar(broadcast_var + self.eps, 1/2) + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        self.w = Parameter(init.ones(dim, device=device, dtype=dtype), dtype=dtype, device=device)
        self.b = Parameter(init.zeros(dim, device=device, dtype=dtype), dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        E = ops.broadcast_to(ops.reshape(ops.summation(x, 1), (x.shape[0], 1)), x.shape) / self.dim
        V = ops.broadcast_to(ops.reshape(ops.summation((x - E) ** 2, 1), (x.shape[0], 1)), x.shape) / self.dim
        V_eps = V + self.eps

        return ops.broadcast_to(ops.reshape(self.w, (1, self.dim)), x.shape) * (x - E) / ops.power_scalar(V_eps, 1/2) + ops.broadcast_to(ops.reshape(self.b, (1, self.dim)), x.shape)



class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=(1-self.p), device=x.device, dtype=x.dtype) / (1-self.p)
            return x * mask
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
