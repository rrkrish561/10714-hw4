"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for (i, param) in enumerate(self.params):
            if param.grad is None:
                continue

            if self.u.get(i) is None:
                self.u[i] = ndl.zeros_like(param)

            self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * (param.grad.data + self.weight_decay * param.data)

            param.data -= ndl.Tensor(self.lr * self.u[i], dtype=param.dtype)
            

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

        for (i, param) in enumerate(self.params):
            if param.grad is None:
                continue

            if self.m.get(i) is None:
                self.m[i] = ndl.zeros_like(param.data)
                self.v[i] = ndl.zeros_like(param.data)

            grad = param.grad.data + self.weight_decay * param.data

            self.m[i].data = ndl.Tensor(self.beta1 * self.m[i].data + (1 - self.beta1) * (grad.data), dtype=param.dtype)
            self.v[i].data = ndl.Tensor(self.beta2 * self.v[i].data + (1 - self.beta2) * ((grad.data) ** 2), dtype=param.dtype)

            m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[i].data / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat.data / (ndl.power_scalar(v_hat.data, 1/2) + self.eps)
