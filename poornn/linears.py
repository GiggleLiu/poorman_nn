'''
Linear Layer.
'''

import numpy as np
import scipy
import scipy.sparse as sps
import pdb

from .core import Layer, EMPTY_VAR
from .lib.spsp import lib as fspsp
from .lib.linear import lib as flinear
from .utils import masked_concatenate, dtype2token,typed_randn

__all__=['LinearBase', 'Linear', 'SPLinear', 'Apdot']

class LinearBase(Layer):
    '''
    Linear Layer.

    Attributes:
        :input_shape: two types allowed, (num_batch, weight.shape[1]), or (weight.shape[1],)
        :weight: 2darray, (fout, fin), in fortran order.
        :bias: 1darray/None, bias of shape (fout,), zeros if None.
    '''
    __display_attrs__ = ['var_mask']
    def __init__(self, input_shape, itype, weight, bias, var_mask=(1,1)):
        if sps.issparse(weight):
            self.weight = weight.tocsr()
        else:
            self.weight = np.asarray(weight, order='F')
        if bias is None or bias is 0:
            bias = np.zeros(weight.shape[0],dtype=weight.dtype)
        else:
            bias = np.asarray(bias)
        self.bias = bias
        output_shape = input_shape[:-1]+(weight.shape[0],)
        if len(var_mask)!=2:
            raise ValueError('length of mask error, expect 2, but get %s!'%len(var_mask))
        self.var_mask = var_mask
        super(LinearBase, self).__init__(input_shape, output_shape, itype=itype, dtype=np.find_common_type((weight.dtype,bias.dtype),()).name)

    def get_variables(self):
        dvar=masked_concatenate([self.weight.ravel(order='F') if not sps.issparse(self.weight) else self.weight.data, self.bias], self.var_mask)
        return dvar

    def set_variables(self, variables):
        nw=self.weight.size if self.var_mask[0] else 0
        var1, var2 = variables[:nw], variables[nw:]
        weight_data = self.weight.data if sps.issparse(self.weight) else self.weight.ravel(order='F')
        if self.var_mask[0]: weight_data[:] = var1
        if self.var_mask[1]: self.bias[:] = var2

    @property
    def num_variables(self):
        return (self.weight.size if self.var_mask[0] else 0)+(self.bias.size if self.var_mask[1] else 0)

class Linear(LinearBase):
    '''
    Dense Linear Layer, f = x.dot(weight.T) + bias
    '''
    __display_attrs__ = ['var_mask','is_unitary']

    def __init__(self, input_shape, itype, weight, bias, var_mask=(1,1), is_unitary=False, **kwargs):
        if isinstance(weight,tuple):
            weight = 0.1*typed_randn(kwargs.get('dtype',itype),weight)
        if input_shape[-1] != weight.shape[1]:
            raise ValueError('Shape Mismatch!')
        super(Linear, self).__init__(input_shape, itype=itype, weight=weight, bias=bias, var_mask=var_mask)

        dtype_token = dtype2token(np.find_common_type((self.itype,self.dtype),()))
        self._fforward=eval('flinear.forward_%s'%(dtype_token))
        self._fbackward=eval('flinear.backward_%s'%(dtype_token))

        # make it unitary
        self.is_unitary = is_unitary
        if is_unitary:
            self.be_unitary()
            self.check_unitary()

    def forward(self, x, **kwargs):
        y = self._fforward(np.atleast_2d(x), self.weight, self.bias)
        return y.reshape(self.output_shape, order='F')

    def backward(self, xy, dy, **kwargs):
        mask = self.var_mask
        x,y = xy
        dx, dweight, dbias = self._fbackward(np.atleast_2d(dy), np.atleast_2d(x), self.weight,
            do_xgrad=True, do_wgrad=mask[0], do_bgrad=mask[1])
        dvar=masked_concatenate([dweight.ravel(order='F'), dbias], mask)
        return dvar, dx.reshape(self.input_shape, order='F')

    def be_unitary(self):
        self.weight = np.linalg.qr(self.weight.T)[0].T
        self.is_unitary = True

    def check_unitary(self, tol=1e-10):
        weight = self.weight
        # check weight shape
        if self.weight.shape[1]<self.weight.shape[0]:
            raise ValueError('output shape greater than input shape error!')

        # get unitary error
        err = abs(weight.dot(weight.T.conj())-np.eye(weight.shape[0])).mean()
        if self.is_unitary and err>tol:
            raise ValueError('non-unitary matrix error, error = %s!'%err)
        return err

    def set_variables(self, variables):
        nw=self.weight.size if self.var_mask[0] else 0
        var1, var2 = variables[:nw], variables[nw:]
        weight_data = self.weight.data if sps.issparse(self.weight) else self.weight.ravel(order='F')
        if self.is_unitary and self.var_mask[0]:
            W = self.weight
            dG = var1.reshape(W.shape,order='F') - W
            #dG = dG.conj()
            dA = W.T.conj().dot(dG) - dG.T.conj().dot(W)
            #dA=dA.conj()
            B = np.eye(dG.shape[1]) - dA/2
            Y = W.dot(B.T.conj()).dot(np.linalg.inv(B))
            self.weight[...] = Y
        elif self.var_mask[0]:
            weight_data[:] = var1
        if self.var_mask[1]:
            self.bias[:] = var2


class Apdot(LinearBase):
    '''product layer.'''

    def forward(self, x, **kwargs):
        if x.ndim==1:
            x=x[np.newaxis]
        y=(np.prod(self.weight+x[:,np.newaxis,:],axis=2)*self.bias).reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
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
    def __init__(self, input_shape, itype, weight, bias, strides=None, var_mask=(1,1), **kwargs):
        if input_shape[-1] != weight.shape[1]:
            raise ValueError('Shape Mismatch!')
        super(SPLinear, self).__init__(input_shape, itype=itype, weight=weight, bias=bias, var_mask=var_mask)

        dtype_token = dtype2token(np.find_common_type((self.itype,self.dtype),()))
        self._fforward=eval('fspsp.forward%s'%(dtype_token))
        self._fbackward=eval('fspsp.backward%s'%(dtype_token))

    def forward(self, x, **kwargs):
        y = self._fforward(np.atleast_2d(x), csc_indices=self.weight.indices+1, csc_indptr=self.weight.indptr+1,\
                csc_data=self.weight.data, bias=self.bias)
        return y.reshape(self.output_shape, order='F')

    def backward(self, xy, dy, **kwargs):
        x,y = xy
        mask = self.var_mask
        dx, dweight, dbias = self._fbackward(np.atleast_2d(dy), np.atleast_2d(x), csc_data=self.weight.data, csc_indices=self.weight.indices+1,\
            csc_indptr=self.weight.indptr+1, do_xgrad=True, do_wgrad=mask[0], do_bgrad=mask[1])

        dvar=masked_concatenate([dweight.ravel(order='F'), dbias], mask)
        return dvar, dx.reshape(self.input_shape, order='F')
