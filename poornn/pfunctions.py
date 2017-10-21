import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre,\
    Laguerre, Hermite, HermiteE
from scipy.misc import factorial
import pdb

from .core import ParamFunction, EMPTY_VAR
from .utils import fsign, dtype_c2r, dtype_r2c

__all__ = ['PReLU', 'Poly', 'Mobius', 'Georgiou1992', 'Gaussian', 'PMul']


class PReLU(ParamFunction):
    '''
    Parametric ReLU,

    .. math::
        :nowrap:

        \\begin{align}
        f(x)=\\text{relu}(x)=\\begin{cases}
                x, &\Re[x]>0\\\\
                \\text{leak}\cdot x,&\Re[x]<0
                \end{cases}
        \\end{align}

    where leak is a trainable parameter if var_mask[0] is True.

    Args:
        leak (float, default=0.1): leakage,
        var_mask (1darray<bool>, default=[True]): variable mask

    Attributes:
        leak (float): leakage,
        var_mask (1darray<bool>): variable mask
    '''
    __display_attrs__ = ['leak']

    def __init__(self, input_shape, itype, leak=0.1, var_mask=[True]):
        dtype = np.find_common_type(('float32', np.dtype(type(leak))), ()).name
        otype = np.find_common_type((dtype, itype), ()).name
        if leak > 1 or leak < 0:
            raise ValueError('leak parameter should be 0-1!')
        super(PReLU, self).__init__(input_shape, input_shape,
                                    itype, otype=otype,
                                    dtype=dtype, params=[leak],
                                    var_mask=var_mask)

    @property
    def leak(self): return self.params[0]

    def forward(self, x, **kwargs):
        if self.leak == 0:
            return np.maximum(x, 0)
        else:
            return np.maximum(x, self.leak * x)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        dx = dy.copy(order='F')
        xmask = x < 0
        if self.leak == 0:
            dx[xmask] = 0
        else:
            dx[xmask] = self.leak * dy[xmask]
        if self.var_mask[0]:
            da = np.sum(dy[xmask] * x[xmask].conj())
            dw = np.array([da], dtype=self.dtype)
        else:
            dw = EMPTY_VAR
        return dw, dx


class Poly(ParamFunction):
    '''
    Ploynomial function layer.

        e.g. for polynomial kernel, we have
            * :math:`f(x) = \sum\limits_i \\text{params}[i]x^i/i!` \
(factorial_rescale is True)
            * :math:`f(x) = \sum\limits_i \\text{params}[i]x^i` \
(factorial_rescale is False)

    Args:
        kernel (str, default='polynomial'): the kind of polynomial
            serie expansion, see `Poly.kernel_dict for detail.`
        factorial_rescale (bool, default=False): rescale
            high order factors to avoid overflow.
        var_mask (1darray<bool>, default=(True,True,...)): variable mask

    Attributes:
        kernel (str): the kind of polynomial serie expansion,
            see `Poly.kernel_dict for detail.`
        factorial_rescale (bool): rescale high order factors to avoid overflow.
        var_mask (1darray<bool>): variable mask
    '''
    __display_attrs__ = ['kernel', 'max_order',
                         'var_mask', 'factorial_rescale']

    kernel_dict = {'polynomial': Polynomial, 'chebyshev': Chebyshev,
                   'legendre': Legendre, 'laguerre': Laguerre,
                   'hermite': Hermite, 'hermiteE': HermiteE}
    '''dict of available kernels, with values target functions.'''

    def __init__(self, input_shape, itype, params,
                 kernel='polynomial', var_mask=None,
                 factorial_rescale=False):
        # check input data
        if kernel not in self.kernel_dict:
            raise ValueError('Kernel %s not found, should be one of %s' % (
                kernel, self.kernel_dict))

        super(Poly, self).__init__(input_shape, input_shape,
                                   itype, params=params, var_mask=var_mask)
        self.kernel = kernel
        self.factorial_rescale = factorial_rescale

    @property
    def max_order(self):
        '''int: maximum order appeared.'''
        return len(self.params) - 1

    def forward(self, x, **kwargs):
        factor = 1. / factorial(np.arange(len(self.params))
                                ) if self.factorial_rescale else 1
        p = self.kernel_dict[self.kernel](self.params * factor)
        y = p(x)
        return y

    def backward(self, xy, dy, **kwargs):
        factor = 1. / factorial(np.arange(len(self.params))
                                ) if self.factorial_rescale\
            else np.ones(len(self.params))
        x, y = xy
        p = self.kernel_dict[self.kernel](self.params * factor)
        dp = p.deriv()
        dx = dp(x) * dy
        dw = []
        for i, mask in enumerate(self.var_mask):
            if mask:
                basis_func = self.kernel_dict[self.kernel].basis(i)
                dwi = (basis_func(x) * dy * factor[i]).sum()
                dw.append(dwi)
        return np.array(dw, dtype=self.dtype), dx


class Mobius(ParamFunction):
    '''
    Mobius transformation, :math:`f(x) = \\frac{(z-a)(b-c)}{(z-c)(b-a)}`

    :math:`a, b, c` map to :math:`0, 1, \infty` respectively.
    '''
    __display_attrs__ = ['var_mask']

    def __init__(self, input_shape, itype, params, var_mask=None):
        # check input data
        if len(params) != 3:
            raise ValueError('Mobius take 3 params! but get %s' % len(params))

        super(Mobius, self).__init__(input_shape, input_shape,
                                     itype, params=params, var_mask=var_mask)

    def forward(self, x, **kwargs):
        a, b, c = self.params
        return (b - c) / (b - a) * (x - a) / (x - c)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        a, b, c = self.params
        dx = (a - c) * (c - b) / (a - b) / (x - c)**2 * dy
        dw = []
        if self.var_mask[0]:
            dw.append((b - c) / (a - b)**2 * ((x - b) / (x - c) * dy).sum())
        if self.var_mask[1]:
            dw.append(-(a - c) / (a - b)**2 * ((x - a) / (x - c) * dy).sum())
        if self.var_mask[2]:
            dw.append(((x - a) * (x - b) / (x - c)**2 / (a - b) * dy).sum())
        return np.array(dw, dtype=self.dtype), dx


class Georgiou1992(ParamFunction):
    '''
    Function :math:`f(x) = \\frac{x}{c+|x|/r}`
    '''
    __display_attrs__ = ['c', 'r', 'var_mask']

    def __init__(self, input_shape, itype, params, var_mask=None):
        params = np.asarray(params)
        if np.iscomplexobj(params):
            raise ValueError(
                'Args c, r for %s should not be complex!'
                % self.__class__.__name__)
        if params[1] == 0:
            raise ValueError('r = 0 get!')

        super(Georgiou1992, self).__init__(input_shape, input_shape,
                                           itype, params=params,
                                           var_mask=var_mask,
                                           tags={'analytical': 3})

    @property
    def c(self): return self.params[0]

    @property
    def r(self): return self.params[1]

    def forward(self, x, **kwargs):
        c, r = self.params
        return x / (c + np.abs(x) / r)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        c, r = self.params
        deno = 1. / (c + np.abs(x) / r)**2
        dw = []
        if self.var_mask[0]:
            dw.append((-x * deno * dy).real.sum())
        if self.var_mask[1]:
            dw.append((x * np.abs(x) / r**2 * deno * dy).real.sum())
        dx = c * dy * deno
        if self.otype[:7] == 'complex':
            dx = dx + x.conj() / r * 1j * (fsign(x) * dy).imag * deno
        return np.array(dw, dtype=self.dtype), dx


class Gaussian(ParamFunction):
    '''
    Function :math:`f(x) = \\frac{1}{\sqrt{2\pi}\sigma} \
\exp(-\\frac{\\|x-\mu\\|^2}{2\sigma^2})`,
    where :math:`\mu,\sigma` are mean and variance respectively.
    '''
    __display_attrs__ = ['mean', 'variance', 'var_mask']

    def __init__(self, input_shape, itype, params, var_mask=None):
        dtype = itype
        params = np.asarray(params, dtype=dtype)
        otype = dtype_c2r(itype) if itype[:7] == 'complex' else itype
        if params[1].imag != 0 or params[1] <= 0:
            raise ValueError('non-positive variance get!')

        super(Gaussian, self).__init__(input_shape, input_shape, itype,
                                       dtype=dtype, otype=otype, params=params,
                                       var_mask=var_mask,
                                       tags={'analytical': 2})

    @property
    def mean(self): return self.params[0]

    @property
    def variance(self): return self.params[1]

    def forward(self, x, **kwargs):
        mu, sig = self.params
        sig = np.real(sig)
        xx = x - mu
        return np.exp(-(xx * xx.conj()).real / (2 * sig**2.)) /\
            np.sqrt(2 * np.pi) / sig

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        ydy = y * dy
        mu, sig = self.params
        xx = x - mu
        sig = np.real(sig)
        dw = []
        if self.var_mask[0]:
            dw.append((xx.real / sig**2 * ydy).sum())
        if self.var_mask[1]:
            dw.append(((xx * xx.conj() - sig**2).real / sig**3 * ydy).sum())
        dx = -xx.conj() / sig**2 * ydy
        return np.array(dw, dtype=self.dtype), dx


class PMul(ParamFunction):
    '''
    Function :math:`f(x) = cx`, where c is trainable if var_mask[0] is True.

    Args:
        c (number, default=1.0): multiplier.

    Attributes:
        c (number): multiplier.
    '''
    __display_attrs__ = ['c', 'var_mask']

    def __init__(self, input_shape, itype, c=1., var_mask=None):
        if var_mask is None:
            var_mask = [True]
        params = np.atleast_1d(c)
        dtype = params.dtype.name
        otype = np.find_common_type((dtype, itype), ()).name

        super(PMul, self).__init__(input_shape, input_shape, itype,
                                   dtype=dtype, otype=otype, params=params,
                                   var_mask=np.atleast_1d(var_mask))

    @property
    def c(self): return self.params[0]

    def forward(self, x, **kwargs):
        return self.params[0] * x

    def backward(self, xy, dy, **kwargs):
        c = self.params[0]
        dx = dy * c
        dw = EMPTY_VAR if not self.var_mask[0] else np.array(
            [(dy * xy[0]).sum()])
        return dw, dx
