'''
Linear Layer.
'''

import numpy as np
import scipy.sparse as sps
import pdb

from .core import Layer, EMPTY_VAR
from .lib.spsp import lib as fspsp
from .lib.linear import lib as flinear
from .utils import masked_concatenate

__all__=['LinearBase', 'Linear', 'SPLinear', 'Apdot']

class LinearBase(Layer):
    '''
    Linear Layer.

    Attributes:
        :input_shape: two types allowed, (num_batch, weight.shape[1]), or (weight.shape[1],)
        :weight: 2darray, (fout, fin), in fortran order.
        :bias: 1darray, (fout,)
    '''
    __graphviz_attrs__ = ['var_mask']
    def __init__(self, input_shape, dtype, weight, bias, var_mask=(1,1)):
        if sps.issparse(weight):
            self.weight = weight.tocsr()
        else:
            self.weight = np.asarray(weight, dtype=dtype, order='F')
        self.bias = np.asarray(bias,dtype=dtype)
        output_shape = input_shape[:-1]+(weight.shape[0],)
        self.var_mask = var_mask
        super(LinearBase, self).__init__(input_shape, output_shape, dtype=dtype)

    def __str__(self):
        return self.__repr__()+'\n  - dtype = %s\n  - weight => %s\n  - bias => %s'%(self.dtype,self.weight.shape,self.bias.shape)

    def get_variables(self):
        dvar=masked_concatenate([self.weight.ravel(order='F') if not sps.issparse(self.weight) else self.weight.data, self.bias], self.var_mask)
        return dvar

    def set_variables(self, variables, mode='set'):
        nw=self.weight.size if self.var_mask[0] else 0
        var1, var2 = variables[:nw], variables[nw:]
        weight_data = self.weight.data if sps.issparse(self.weight) else self.weight.ravel(order='F')
        if mode=='set':
            if self.var_mask[0]: np.copyto(weight_data, var1)
            if self.var_mask[1]: np.copyto(self.bias,var2)
        elif mode=='add':
            if self.var_mask[0]: weight_data+=var1
            if self.var_mask[1]: self.bias+=var2

    @property
    def num_variables(self):
        return (self.weight.size if self.var_mask[0] else 0)+(self.bias.size if self.var_mask[1] else 0)

class Linear(LinearBase):
    '''
    Dense Linear Layer.
    '''
    def __init__(self, input_shape, dtype, weight, bias, var_mask=(1,1), **kwargs):
        if input_shape[-1] != weight.shape[1]:
            raise ValueError('Shape Mismatch!')
        super(Linear, self).__init__(input_shape, dtype=dtype, weight=weight, bias=bias, var_mask=var_mask)

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

    def forward(self, x):
        y = self._fforward(np.atleast_2d(x), self.weight, self.bias)
        return y.reshape(self.output_shape, order='F')

    def backward(self, xy, dy):
        mask = self.var_mask
        x,y = xy
        dx, dweight, dbias = self._fbackward(np.atleast_2d(dy), np.atleast_2d(x), self.weight,
            do_xgrad=True, do_wgrad=mask[0], do_bgrad=mask[1])
        dvar=masked_concatenate([dweight.ravel(order='F'), dbias], mask)
        return dvar, dx.reshape(self.input_shape, order='F')

class Apdot(LinearBase):
    '''product layer.'''

    def forward(self, x):
        if x.ndim==1:
            x=x[np.newaxis]
        y=(np.prod(self.weight+x[:,np.newaxis,:],axis=2)*self.bias).reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy):
        x,y = xy
        if dy.ndim==1:
            dy=dy[np.newaxis]
            x=x[np.newaxis]
            y=y[np.newaxis]
        pmat=(dy*y)[:,:,np.newaxis]/(self.weight+x[:,np.newaxis,:])
        dweight=pmat.sum(axis=0)
        dx=pmat.sum(axis=1)
        dbias=((dy*y)/self.bias).sum(axis=0)
        return np.concatenate([dweight.ravel(order='F'), dbias]), dx.reshape(self.input_shape, order='F')

class SPLinear(LinearBase):
    '''
    Attributes:
        :input_shape: (batch, feature_in), or (feature_in,)
        :weight: csr_matrix, with shape (feature_out, feature_in,)
        :bias: 1darray, (feature_out), in fortran order.
        :strides: tuple, displace for convolutions.
    '''
    def __init__(self, input_shape, dtype, weight, bias, strides=None, var_mask=(1,1), **kwargs):
        if input_shape[-1] != weight.shape[1]:
            raise ValueError('Shape Mismatch!')
        super(SPLinear, self).__init__(input_shape, dtype=dtype, weight=weight, bias=bias, var_mask=var_mask)

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
        self._fforward=eval('fspsp.forward%s'%(dtype_token))
        self._fbackward=eval('fspsp.backward%s'%(dtype_token))

    def forward(self, x):
        y = self._fforward(np.atleast_2d(x), csc_indices=self.weight.indices+1, csc_indptr=self.weight.indptr+1,\
                csc_data=self.weight.data, bias=self.bias)
        return y.reshape(self.output_shape, order='F')

    def backward(self, xy, dy):
        x,y = xy
        mask = self.var_mask
        dx, dweight, dbias = self._fbackward(np.atleast_2d(dy), np.atleast_2d(x), csc_data=self.weight.data, csc_indices=self.weight.indices+1,\
            csc_indptr=self.weight.indptr+1, do_xgrad=True, do_wgrad=mask[0], do_bgrad=mask[1])

        dvar=masked_concatenate([dweight.ravel(order='F'), dbias], mask)
        return dvar, dx.reshape(self.input_shape, order='F')

