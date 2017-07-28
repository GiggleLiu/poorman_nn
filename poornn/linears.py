'''
Linear Layer.
'''

import numpy as np
import scipy.sparse as sps
import pdb

from core import Layer
from lib.linear import lib as flinear

__all__=['Linear']

class Linear(Layer):
    '''
    Linear Layer.

    Attributes:
        :weight: 2darray, (fout, fin), in fortran order.
        :bias: 1darray, (fout,)
    '''
    def __init__(self, weight, bias, dtype = 'float32', input_shape=None, output_shape=None):
        self.weight = np.asfortranarray(weight, dtype=dtype)
        self.bias = np.asarray(bias,dtype=dtype)
        if input_shape is None:
            input_shape = (-1,weight.shape[1])
        if output_shape is None:
            output_shape = (-1,weight.shape[0])
        super(Linear, self).__init__(input_shape, output_shape, dtype=dtype)

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

    def __str__(self):
        return self.__repr__()+'\n  dtype = %s\n  weight => %s\n  bias => %s'%(self.dtype,self.weight.shape,self.bias.shape)

    def forward(self, x):
        y = self._fforward(np.atleast_2d(x), self.weight, self.bias)
        return y.reshape(self.output_shape, order='F')

    def backward(self, xy, dy, mask=(1,1)):
        x,y = xy
        dx, dweight, dbias = self._fbackward(np.atleast_2d(dy), np.atleast_2d(x), self.weight,
            do_xgrad=mask[1], do_wgrad=mask[0], do_bgrad=mask[0])
        return np.concatenate([dweight.ravel(order='F'), dbias]), dx.reshape(self.input_shape)

    def get_variables(self):
        return np.concatenate([self.weight.ravel(order='F'),self.bias])

    def set_variables(self, variables, mode='set'):
        if mode=='set':
            np.copyto(self.weight,variables[:self.weight.size].reshape(self.weight.shape, order='F'))
            np.copyto(self.bias,variables[self.weight.size:].reshape(self.bias.shape, order='F'))
        elif mode=='add':
            self.weight+=variables[0]
            self.bias+=variables[1]

    @property
    def num_variables(self):
        return self.weight.size+self.bias.size
