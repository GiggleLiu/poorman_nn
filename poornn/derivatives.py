import numpy as np
import pdb

from . import functions, pfunctions, linears
from .nets import KeepSignFunc, JointComplex
from .utils import dtype_c2r, dtype_r2c

'''
* KS_: nets.KeepSignFunc
* JC_: nets.JointComplex
'''

__all__ = ['KS_Tanh', 'KS_Georgiou1992', 'JC_Tanh', 'JC_Sigmoid', 'JC_Georgiou1992']

def KS_Tanh(input_shape, itype, **kwargs):
    '''Function tanh(|x|)*exp(i theta_x). Hirose 1994'''
    func = functions.Tanh(input_shape, dtype_c2r(itype), **kwargs)
    ks = KeepSignFunc(func)
    return ks

def KS_Georgiou1992(input_shape, itype, cr, var_mask=[False,False], **kwargs):
    '''
    Function f(x) = x/(c+|x|/r)

    Parameters:
        :cr: len-2 tuple, c and r.
        :var_mask: mask for variables (v, w), with v = -c*r and w = -cr/(1-r).
    '''
    c,r = cr
    func = pfunctions.Mobius(input_shape, dtype_c2r(itype), params=np.array([0,-c*r/(1-r),-c*r]), var_mask=[False]+list(var_mask), **kwargs)
    ks = KeepSignFunc(func)
    return ks

def JC_Tanh(input_shape, itype, **kwargs):
    '''Function f(x) = tanh(x.real) + 1j*tanh(x.imag). Kechriotis 1994'''
    func = functions.Tanh(input_shape, dtype_c2r(itype),**kwargs)
    jc = JointComplex(func,func)  # same function
    return jc

def JC_Sigmoid(input_shape, itype, **kwargs):
    '''Function f(x) = sigmoid(x.real) + 1j*sigmoid(x.imag). Birx 1992'''
    func = functions.Sigmoid(input_shape, dtype_c2r(itype),**kwargs)
    jc = JointComplex(func,func)  # same function
    return jc

def JC_Georgiou1992(input_shape, itype, params, **kwargs):
    '''Function f(x) = Georgiou1992(x.real) + 1j*Georgiou1992(x.imag). Kuroe 2005'''
    func = pfunctions.Georgiou1992(input_shape, dtype_c2r(itype), params, var_mask=[False,False], **kwargs)
    jc = JointComplex(func, func)  # same function
    return jc
