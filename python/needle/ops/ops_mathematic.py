"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        if a.shape != b.shape:
          raise ValueError("Shapes must be the same")
        return a ** b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad * b * a ** (b - 1), out_grad * a ** b * log(a)


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        val = node.inputs[0]
        return out_grad * self.scalar * val ** (self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        if a.shape != b.shape:
          raise ValueError("Shapes must be the same")
        return a / b

    def gradient(self, out_grad, node):
        num, den = node.inputs
        return (out_grad / den, out_grad * (-num / den ** 2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.transpose(a, axes=self.axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.compact().reshape(self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        broadcasted_shape = self.shape
        original_shape = node.inputs[0].shape

        original_shape = (1,) * (len(broadcasted_shape) - len(original_shape)) + original_shape
        ax = []
        for i, (o_dim, b_dim) in enumerate(zip(original_shape, broadcasted_shape)):
            if(o_dim == 1 and b_dim > 1):
                ax.append(i)
        return reshape(summation(out_grad, tuple(ax)), node.inputs[0].shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.sum(a)
        elif isinstance(self.axes, Number):
            return array_api.sum(a, axis=(self.axes,))
        else:
            out = a
            for ax in sorted(self.axes, reverse=True):
                out = array_api.sum(out, axis=(ax, ))
            return out

    def gradient(self, out_grad, node):
        new_shape = list(node.inputs[0].shape)
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(new_shape)))
        elif isinstance(self.axes, Number):
            axes = (self.axes,)
        for x in axes:
            new_shape[x] = 1

        return broadcast_to(reshape(out_grad, new_shape), node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        l, r = node.inputs

        r_sum_ax = []
        for i in range(len(l.shape) - len(r.shape)):
            r_sum_ax.append(i)

        l_sum_ax = []
        for i in range(len(r.shape) - len(l.shape)):
            l_sum_ax.append(i)
        return summation(matmul(out_grad, transpose(r)), tuple(l_sum_ax)), summation(matmul(transpose(l), out_grad), tuple(r_sum_ax))



def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        zero = array_api.full(a.shape, 0, device=a.device)
        return array_api.maximum(zero, a)

    def gradient(self, out_grad, node):
        return out_grad * Tensor(node.numpy() > 0, device=node.device, dtype=node.dtype)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        return out_grad - out_grad * node * node


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        # Check if all tensors have the same shape
        shape = args[0].shape
        for arg in args:
            if arg.shape != shape:
                raise ValueError("All tensors must have the same shape")
            
        stacked_shape = shape[:self.axis] + (len(args),) + shape[self.axis:]

        stacked_array = array_api.empty(stacked_shape, dtype=args[0].dtype, device=args[0].device)
        slices = [(None, None, None)] * len(stacked_shape)
        slices[self.axis] = (0, 1, None)
        for i, arg in enumerate(args):
            slice_objects = [slice(*s) for s in slices]
            stacked_array[tuple(slice_objects)] = arg

            slices[self.axis] = (slices[self.axis][0] + 1, slices[self.axis][1] + 1, None)

        return stacked_array

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        num_splits = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)

        split_tensors = []

        for i in range(num_splits):
            slice_objects = [slice(None)] * len(A.shape)
            slice_objects[self.axis] = i
            next_slice = A[tuple(slice_objects)].compact().reshape(new_shape)
            split_tensors.append(next_slice)

        return tuple(split_tensors)

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # Get new shape
        new_shape = list(a.shape)
        for ax in self.axes:
            if ax >= len(new_shape):
                return a
            new_shape[ax] *= self.dilation + 1

        # Create new array
        dilated = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)

        # Fill in values
        slices = [(0, a , 1) for a in new_shape]

        for ax in self.axes:
            slices[ax] = (0, new_shape[ax], self.dilation + 1)

        dilated[tuple([slice(*s) for s in slices])] = a

        return dilated
        

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # Get new shape
        new_shape = list(a.shape)
        for ax in self.axes:
            if ax >= len(new_shape):
                return a
            new_shape[ax] //= self.dilation + 1

        # Create new array
        undilated = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)

        # Fill in values
        slices = [(0, a , 1) for a in new_shape]

        for ax in self.axes:
            slices[ax] = (0, a.shape[ax], self.dilation + 1)

        undilated = a[tuple([slice(*s) for s in slices])]

        return undilated

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        pad_axes = [(0, 0)] * len(A.shape)
        pad_axes[1] = (self.padding, self.padding)
        pad_axes[2] = (self.padding, self.padding)
        A_pad = A.pad(tuple(pad_axes))
        N, H, W, C_in = A_pad.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A_pad.strides

        inner_dim = K * K * C_in

        Z = A_pad.as_strided(shape=(N, (H-K)//self.stride+1, (W-K)//self.stride+1, K, K, C_in), strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs))
        Z = Z.compact().reshape((N * ((H-K)//self.stride+1) * ((W-K)//self.stride+1), inner_dim))
    
        out = Z @ B.compact().reshape((inner_dim, C_out))
        out = out.compact().reshape((N, (H-K)//self.stride+1, (W-K)//self.stride+1, C_out))

        return out

    def gradient(self, out_grad, node):
        A, B = node.inputs
        N, H, W, _ = A.shape
        K, _, _, _ = B.shape

        B_flipped = flip(B, (0, 1)).transpose()
        out_grad_dilate = dilate(out_grad, (1, 2), self.stride - 1)
        grad_A = conv(out_grad_dilate, B_flipped, padding=K-self.padding-1)

        A_permute = A.transpose((0, 3))

        out_grad_permute = out_grad.transpose((0, 1)).transpose((1, 2))
        out_grad_dilate = dilate(out_grad_permute, (0, 1), self.stride - 1)

        grad_B = conv(A_permute, out_grad_dilate, padding=self.padding)
        grad_B = grad_B.transpose((0, 1)).transpose((1, 2))

        return grad_A, grad_B


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


