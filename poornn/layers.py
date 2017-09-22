import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Laguerre, Hermite, HermiteE
import pdb

from .core import Layer

__all__=['PReLU', 'Poly']

class PReLU(Layer):
    '''
    Parametric ReLU.
    '''
    __display_attrs__ = ['leak']

    def __init__(self, input_shape, itype, leak = 0):
        self.leak = leak
        self.itype=itype
        if leak>1 or leak<0:
            raise ValueError('leak parameter should be 0-1!')
        super(PReLU,self).__init__(input_shape, input_shape, itype, otype=itype,
                dtype=np.dtype(type(leak)).name)

    def __call__(self,x):
        return self.forward(x)

    def forward(self, x):
        if self.leak==0:
            return maximum(x,0)
        else:
            return maximum(x,self.leak*x)

    def backward(self, x, y, dy, **kwargs):
        dx=dy.copy(order='F')
        xmask=x<0
        if self.leak==0:
            dx[xmask]=0
        else:
            dx[xmask]=leak*dy
        da = np.sum(dy[xmask]*x[xmask].conj())
        return np.array([da], dtype=self.itype), dx

    def get_variables(self):
        return np.array([self.leak], dtype=self.dtype)

    def set_variables(self, a):
        self.leak=a[0]

    @property
    def num_variables(self):
        return 1

class Poly(Layer):
    '''
    Ploynomial function layer.

        e.g. for polynomial kernel, we have f(x) = \sum_i params[i]*x^i
    '''
    __display_attrs__ = ['kernel', 'max_order', 'var_mask']
    kernel_dict = {'polynomial':Polynomial,'chebyshev':Chebyshev,
            'legendre':Legendre,'laguerre':Laguerre,'hermite':Hermite,'hermiteE':HermiteE}

    def __init__(self, input_shape, itype, params, kernel='polynomial', var_mask=None):
        # check input data
        params = np.asarray(params)
        if var_mask is None:
            var_mask = np.ones(len(params),dtype='bool')
        else:
            var_mask = np.asarray(var_mask, dtype='bool')
        if kernel not in self.kernel_dict:
            raise ValueError('Kernel %s not found, should be one of %s'%(kernel,self.kernel_dict))

        dtype = params.dtype.name
        otype = np.find_common_type((dtype, itype),())
        super(Poly,self).__init__(input_shape, input_shape, itype, otype=otype, dtype=dtype)
        self.params = params
        self.kernel = kernel
        self.var_mask = var_mask

    @property
    def max_order(self):
        return len(self.params)-1

    def forward(self, x, **kwargs):
        p = self.kernel_dict[self.kernel](self.params)
        y = p(x)
        return y

    def backward(self, xy, dy, **kwargs):
        x,y = xy
        p = self.kernel_dict[self.kernel](self.params)
        dp = p.deriv()
        dx = dp(x)*dy
        dw = []
        for i,mask in enumerate(self.var_mask):
            if mask:
                basis_func = self.kernel_dict[self.kernel].basis(i)
                dwi = (basis_func(x)*dy).sum()
                dw.append(dwi)
        return np.array(dw, dtype=self.dtype), dx

    def get_variables(self):
        return self.params[self.var_mask]

    def set_variables(self, a):
        self.params[self.var_mask] = a

    @property
    def num_variables(self):
        return self.var_mask.sum()
