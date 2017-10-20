'''
Derived functions, name prefix specifies its container, like
    * KS\_: nets.KeepSignFunc
    * JC\_: nets.JointComplex
'''

import numpy as np
import pdb

from . import functions, pfunctions, linears
from .nets import KeepSignFunc, JointComplex
from .utils import dtype_c2r, dtype_r2c

__all__ = ['KS_Tanh', 'KS_Georgiou1992',
           'JC_Tanh', 'JC_Sigmoid', 'JC_Georgiou1992']


def KS_Tanh(input_shape, itype, **kwargs):
    '''
    Function :math:`f(x) = \\tanh(|x|)\\exp(i \\theta_x)`.

    References:
        Hirose 1994

    Returns:
        KeepSignFunc: keep sign tanh layer.
    '''
    func = functions.Tanh(input_shape, dtype_c2r(itype), **kwargs)
    ks = KeepSignFunc(func)
    return ks


def KS_Georgiou1992(input_shape, itype, cr, var_mask=[False, False], **kwargs):
    '''
    Function :math:`f(x) = \\frac{x}{c+|x|/r}`

    Args:
        cr (tuplei, len=2): c and r.
        var_mask (1darray, len=2, default=[False,False]):\
                mask for variables (v, w), with v = -c*r and w = -cr/(1-r).

    Returns:
        KeepSignFunc: keep sign Georgiou's layer.
    '''
    c, r = cr
    func = pfunctions.Mobius(input_shape, dtype_c2r(itype), params=np.array(
        [0, -c * r / (1 - r), -c * r]), var_mask=[False] + list(var_mask),
        **kwargs)
    ks = KeepSignFunc(func)
    return ks


def JC_Tanh(input_shape, itype, **kwargs):
    '''
    Function :math:`f(x) = \\tanh(\\Re[x]) + i\\tanh(\\Im[x])`.

    References:
        Kechriotis 1994

    Returns:
        JointComplex: joint complex tanh layer.
    '''
    func = functions.Tanh(input_shape, dtype_c2r(itype), **kwargs)
    jc = JointComplex(func, func)  # same function
    return jc


def JC_Sigmoid(input_shape, itype, **kwargs):
    '''
    Function :math:`f(x) = \\sigma(\\Re[x]) + i\\sigma(\\Im[x])`.

    References:
        Birx 1992

    Returns:
        JointComplex: joint complex sigmoid layer.
    '''
    func = functions.Sigmoid(input_shape, dtype_c2r(itype), **kwargs)
    jc = JointComplex(func, func)  # same function
    return jc


def JC_Georgiou1992(input_shape, itype, params, **kwargs):
    '''
    Function :math:`f(x) = \\text{Georgiou1992}\
            (\\Re[x]) + i\\text{Georgiou1992}(\\Im[x])`.

    Args:
        params: params for Georgiou1992.

    References:
        Kuroe 2005

    Returns:
        JointComplex: joint complex Geogiou's layer.
    '''
    func = pfunctions.Georgiou1992(input_shape, dtype_c2r(
        itype), params, var_mask=[False, False], **kwargs)
    jc = JointComplex(func, func)  # same function
    return jc
