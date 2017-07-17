'''
Linear Layer.
'''

import numpy as np
import scipy.sparse as sps
import pdb

from core import Layer
from lib.linear import lib as flinear

__all__=['L_Tensor', 'Linear']

class L_Tensor(Layer):
    '''
    Tensor Layer.
    
    Attributes:
        :W: ndarray, the weight tensor.
        :einsum_tokens: list, tokens for [x,W,y] in einsum.

    Note:
        dx=W*dy, need conjugate during BP?
    '''
    def __init__(self,W,einsum_tokens):
        self.W=np.asarray(W)
        if len(einsum_tokens)!=3:
            raise ValueError('einsum_tokens should be a len-3 list!')
        if len(einsum_tokens[1])!=self.W.ndim:
            raise ValueError('einsum_tokens dimension error!')
        self.einsum_tokens=einsum_tokens

    def forward(self,x):
        return np.einsum('%s,%s->%s'%tuple(self.einsum_tokens),x,self.W)

    def backward(self,x,y,dy):
        einsum_tokens=self.einsum_tokens
        dx=np.einsum('%s,%s->%s'%tuple(einsum_tokens[::-1]),dy,self.W)
        dW=np.einsum('%s,%s->%s'%(einsum_tokens[0],einsum_tokens[2],einsum_tokens[1]),x,dy)
        return dW,dx

    def get_variables(self):
        return self.W.ravel()

    def set_variables(self,variables):
        if isinstance(variables,np.ndarray):
            variables=variables.reshape(self.W.shape)
        self.W[...]=variables

class Linear(Layer):
    '''
    Linear Layer.

    Attributes:
        :weight: 2darray, (fout, fin), in fortran order.
        :bias: 1darray, (fout,)
    '''
    def __init__(self, weight, bias, dtype = 'float32'):
        self.weight = np.asfortranarray(weight)
        self.bias = bias
        self.dtype = dtype

        if dtype=='complex128':
            dtype_token = 'z'
        elif dtype=='complex64':
            dtype_token = 'c'
        elif dtype=='float64':
            dtype_token = 'd'
        elif dtype=='float32':
            dtype_token = 's'
        else:
            raise TypeError("dtype error!")
        self._fforward=eval('flinear.forward_%s'%(dtype_token))
        self._fbackward=eval('flinear.backward_%s'%(dtype_token))
        #self._fforward1=eval('flinear.forward1_%s'%(dtype_token))
        #self._fbackward1=eval('flinear.backward1_%s'%(dtype_token))


    def forward(self, x):
        y = self._fforward(x, self.weight, self.bias)
        return y

    def backward(self, x, y, dy, dx=None, dweight=None, dbias=None, mask=(1,)*3):
        if dx is None and mask[0]: dx=np.zeros_like(x)
        if dweight is None and mask[1]: dweight=np.zeros_like(self.weight)
        if dbias is None and mask[2]: dbias=np.zeros_like(self.bias)
        y = self._fbackward(dy, x, dx, dweight, dbias, self.weight, self.bias,
            do_xgrad=mask[0], do_wgrad=mask[1], do_bgrad=mask[2])
        return (dweight, dbias), dx

    def get_variables(self):
        return (self.weight,self.bias)

    def set_variables(self, variables, mode='set'):
        if mode=='set':
            self.weight[...]=variables[0]
            self.bias[...]=variables[1]
        elif mode=='add':
            self.weight+=variables[0]
            self.bias+=variables[1]

    @property
    def num_variables(self):
        return 2
