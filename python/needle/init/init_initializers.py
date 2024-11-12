import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))

    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))

    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    
    if shape is not None:
        K, _, I, _ = shape
        a = math.sqrt(6.0 / (I*K*K))

        return rand(*shape, low=-a, high=a, **kwargs)

    a = math.sqrt(6.0 / fan_in)

    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    
    std = math.sqrt(2.0 / fan_in)

    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)