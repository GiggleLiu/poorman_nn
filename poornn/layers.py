import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Laguerre, Hermite, HermiteE
from scipy.misc import factorial
import pdb

from .core import Layer
from .utils import fsign

__all__=['PReLU', 'Poly', 'Mobius', 'Georgiou1992']

class PReLU(Layer):
    '''
    Parametric ReLU.
    '''
    __display_attrs__ = ['leak']

    def __init__(self, input_shape, itype, leak = 0):
        self.leak = leak
        self.itype=itype
        otype = np.find_common_type((dtype, itype),()).name
        if leak>1 or leak<0:
            raise ValueError('leak parameter should be 0-1!')
        super(PReLU,self).__init__(input_shape, input_shape, itype, otype=otype,
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

        e.g. for polynomial kernel, we have 
            * f(x) = \sum_i params[i]*x^i/i!  (factorial_rescale == True)
            * f(x) = \sum_i params[i]*x^i  (factorial_rescale == False)
    '''
    __display_attrs__ = ['kernel', 'max_order', 'var_mask', 'factorial_rescale']
    kernel_dict = {'polynomial':Polynomial,'chebyshev':Chebyshev,
            'legendre':Legendre,'laguerre':Laguerre,'hermite':Hermite,'hermiteE':HermiteE}

    def __init__(self, input_shape, itype, params, kernel='polynomial', var_mask=None, factorial_rescale=False):
        # check input data
        params = np.asarray(params)
        if var_mask is None:
            var_mask = np.ones(len(params),dtype='bool')
        else:
            var_mask = np.asarray(var_mask, dtype='bool')
        if kernel not in self.kernel_dict:
            raise ValueError('Kernel %s not found, should be one of %s'%(kernel,self.kernel_dict))

        dtype = params.dtype.name
        otype = np.find_common_type((dtype, itype),()).name
        super(Poly,self).__init__(input_shape, input_shape, itype, otype=otype, dtype=dtype)
        self.params = params
        self.kernel = kernel
        self.var_mask = var_mask
        self.factorial_rescale = factorial_rescale

    @property
    def max_order(self):
        return len(self.params)-1

    def forward(self, x, **kwargs):
        factor = 1./factorial(np.arange(len(self.params))) if self.factorial_rescale else 1
        p = self.kernel_dict[self.kernel](self.params*factor)
        y = p(x)
        return y

    def backward(self, xy, dy, **kwargs):
        factor = 1./factorial(np.arange(len(self.params))) if self.factorial_rescale else np.ones(len(self.params))
        x,y = xy
        p = self.kernel_dict[self.kernel](self.params*factor)
        dp = p.deriv()
        dx = dp(x)*dy
        dw = []
        for i,mask in enumerate(self.var_mask):
            if mask:
                basis_func = self.kernel_dict[self.kernel].basis(i)
                dwi = (basis_func(x)*dy*factor[i]).sum()
                dw.append(dwi)
        return np.array(dw, dtype=self.dtype), dx

    def get_variables(self):
        return self.params[self.var_mask]

    def set_variables(self, a):
        self.params[self.var_mask] = a

    @property
    def num_variables(self):
        return self.var_mask.sum()

class Mobius(Layer):
    '''
    Mobius transformation, f(x) = (z-a)(b-c)/(z-c)(b-a)
    
    a, b, c map to 0, 1 Inf respectively.
    '''
    __display_attrs__ = ['var_mask']

    def __init__(self, input_shape, itype, params, var_mask=None):
        # check input data
        if len(params)!=3:
            raise ValueError('Mobius take 3 params! but get %s'%len(params))
        params = np.asarray(params)
        if var_mask is None:
            var_mask = np.ones(len(params),dtype='bool')
        else:
            var_mask = np.asarray(var_mask, dtype='bool')

        dtype = params.dtype.name
        otype = np.find_common_type((dtype, itype),()).name
        super(Mobius,self).__init__(input_shape, input_shape, itype, otype=otype, dtype=dtype)
        self.params = params
        self.var_mask = var_mask

    def forward(self, x, **kwargs):
        a,b,c = self.params
        return (b-c)/(b-a)*(x-a)/(x-c)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        a,b,c = self.params
        dx = (a-c)*(c-b)/(a-b)/(x-c)**2*dy
        dw = []
        if self.var_mask[0]:
            dw.append((b-c)/(a-b)**2*((x-b)/(x-c)*dy).sum())
        if self.var_mask[1]:
            dw.append(-(a-c)/(a-b)**2*((x-a)/(x-c)*dy).sum())
        if self.var_mask[2]:
            dw.append(((x-a)*(x-b)/(x-c)**2/(a-b)*dy).sum())
        return np.array(dw, dtype=self.dtype), dx

    def get_variables(self):
        return self.params[self.var_mask]

    def set_variables(self, a):
        self.params[self.var_mask] = a

    @property
    def num_variables(self):
        return self.var_mask.sum()

class Georgiou1992(Layer):
    '''
    Function f(x) = x/(c+|x|/r)
    '''
    __display_attrs__ = ['c', 'r', 'var_mask']

    def __init__(self, input_shape, itype, params, var_mask=None):
        params = np.asarray(params)
        if np.iscomplexobj(params):
            raise ValueError('Parameters c, r for %s should not be complex!'%self.__class__.__name__)
        if params[1] == 0:
            raise ValueError('r = 0 get!')
        if var_mask is None:
            var_mask = np.ones(len(params),dtype='bool')
        else:
            var_mask = np.asarray(var_mask, dtype='bool')

        dtype = params.dtype.name
        otype = np.find_common_type((dtype, itype),()).name
        super(Georgiou1992,self).__init__(input_shape, input_shape, itype, otype=otype, dtype=dtype, tags = {'analytical':3})
        self.params = params
        self.var_mask = var_mask

    @property
    def c(self): return self.params[0]
    @property
    def r(self): return self.params[1]

    def forward(self, x, **kwargs):
        c, r = self.params
        return x/(c+np.abs(x)/r)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        c, r = self.params
        deno = 1./(c+np.abs(x)/r)**2
        dc, dr = -x*deno*dy, x*np.abs(x)/r**2*deno*dy
        dx = c*dy*deno
        if self.otype[:7]=='complex':
            dx = dx + x.conj()/r*1j*(fsign(x)*dy).imag*deno
        return np.array([dc.sum().real,dr.sum().real], dtype=self.dtype), dx

    def get_variables(self):
        return self.params[self.var_mask]

    def set_variables(self, a):
        self.params[self.var_mask] = a

    @property
    def num_variables(self):
        return self.var_mask.sum()
