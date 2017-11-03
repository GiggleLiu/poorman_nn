import numpy as np
import scipy
from scipy import fftpack
from numbers import Number
import pdb

from .core import Layer, Function, EXP_OVERFLOW, EMPTY_VAR
from .lib.pooling import lib as fpooling
from .lib.convprod import lib as fconvprod
from .lib.relu import lib as frelu
from .utils import scan2csc, tuple_prod, dtype2token,\
    dtype_c2r, dtype_r2c, complex_backward, fsign

__all__ = ['wrapfunc', 'Log2cosh', 'Logcosh', 'Sigmoid',
           'Cosh', 'Sinh', 'Tan', 'Tanh',
           'Sum', 'Mul', 'Mod', 'Mean', 'FFT', 'ReLU', 'ConvProd',
           'Pooling', 'DropOut',
           'Sin', 'Cos', 'ArcTan', 'Exp', 'Log', 'SoftPlus', 'Power',
           'SoftMax', 'CrossEntropy', 'SoftMaxCrossEntropy', 'SquareLoss',
           'Reshape', 'Transpose', 'TypeCast', 'Reverse',
           'Filter', 'BatchNorm', 'Normalize',
           'Real', 'Imag', 'Conj', 'Abs', 'Abs2', 'Angle']


def wrapfunc(func, dfunc, classname='GeneralFunc', attrs={},
             docstring="", tags={}, real_out=False):
    '''
    wrap a function and its backward counterpart into \
a :class:`poornn.core.Function` layer.

    Args:
        func (func): forward function, take input (x, attrs) as parameters.
        dfunc (func): derivative function, \
take input/output (x, y, \*\*attrs) as parameters.
        classname (str): function classname,
        attrs (dict): attributes, and default input parameters.
        docstring (str): the docstring of new class.
        tags (dict): tags for this function, \
see `poornn.core.TAG_LIST` for detail.
        real_out (bool): output data type is real for \
any input data type if True.

    Returns:
        class: a dynamically generated layer type.
    '''
    # Parse and validate the field names.  Validation serves two purposes,
    # generating informative error messages and preventing template
    # injection attacks.
    field_names = tuple(map(str, attrs.keys()))
    for name in (classname,) + field_names:
        if not all(c.isalnum() or c == '_' for c in name):
            raise ValueError('Type names and field names\
                             can only contain alphanumeric\
                             characters and underscores: % r' % name)
        if name[0].isdigit():
            raise ValueError('Type names and field names cannot start with\
                             a number: % r' % name)
    seen_names = set()
    for name in field_names:
        if name.startswith('_'):
            raise ValueError(
                'Field names cannot start with an underscore: %r' % name)
        if name in seen_names:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen_names.add(name)

    def __init__(self, input_shape, itype, **kwargs):
        for fieldname in field_names:
            if fieldname in kwargs:
                setattr(self, fieldname, kwargs.pop(fieldname))
            else:
                val = attrs.pop(fieldname)
                if val is None:
                    raise KeyError('You must specify %s' % fieldname)
                else:
                    setattr(self, fieldname, val)
        Function.__init__(self, input_shape, input_shape, itype, tags=tags,
                          otype=kwargs.pop('otype',
                                           dtype_c2r(itype)
                                           if itype[: 5] != 'float' and
                                           real_out else itype), **kwargs)

    def forward(self, x, **kwargs):
        return func(x, **dict([(attr, getattr(self, attr))
                               for attr in attrs]))

    def backward(self, xy, dy, **kwargs):
        return EMPTY_VAR, dfunc(xy, dy,
                                **dict([(attr, getattr(self, attr))
                                        for attr in attrs]))

    newclass = type(classname, (Function,), {
        '__init__': __init__,
        'forward': classmethod(forward) if len(attrs) == 0 else forward,
        'backward': classmethod(backward) if len(attrs) == 0 else backward,
        '__doc__': '%s' % docstring,
    })
    newclass.__display_attrs__ = field_names
    return newclass


class Log2cosh(Function):
    '''
    Function :math:`f(x)=\log(2\cosh(x))`.
    '''

    def __init__(self, input_shape, itype, **kwargs):
        super(Log2cosh, self).__init__(
            input_shape, input_shape, itype, **kwargs)

    @classmethod
    def forward(self, x, **kwargs):
        if np.ndim(x) == 0:
            return scipy.log(2 * np.cosh(x)) if abs(x.real) <= 12\
                else np.sign(x.real) * x
        x = np.asarray(x)
        res = np.zeros_like(x)
        m1 = x.real > EXP_OVERFLOW
        m2 = x.real < -EXP_OVERFLOW
        m3 = ~(m1 | m2)
        res[m1] = x[m1]
        res[m2] = -x[m2]
        res[m3] = scipy.log(2 * np.cosh(x[m3]))
        return res

    @classmethod
    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, np.tanh(x) * dy


class Logcosh(Function):
    '''
    Function :math:`f(x)=\log(\cosh(x))`.
    '''

    def __init__(self, input_shape, itype, **kwargs):
        super(Logcosh, self).__init__(
            input_shape, input_shape, itype, **kwargs)

    @classmethod
    def forward(self, x, **kwargs):
        if np.ndim(x) == 0:
            return scipy.log(np.cosh(x)) if abs(x.real) <= 12\
                else np.sign(x.real) * x
        x = np.asarray(x)
        res = np.zeros_like(x)
        m1 = x.real > EXP_OVERFLOW
        m2 = x.real < -EXP_OVERFLOW
        m3 = ~(m1 | m2)
        res[m1] = x[m1] - scipy.log(2)
        res[m2] = -x[m2] - scipy.log(2)
        res[m3] = scipy.log(np.cosh(x[m3]))
        return res

    @classmethod
    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, np.tanh(x) * dy


class Sigmoid(Function):
    '''
    Function :math:`f(x)=\\frac{1}{1+\exp(-x)}`
    '''

    def __init__(self, input_shape, itype, **kwargs):
        super(Sigmoid, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self, x, **kwargs):
        # for ndarray
        y = np.zeros_like(x)
        m1 = x.real < -EXP_OVERFLOW
        m2 = x.real > EXP_OVERFLOW
        m3 = ~(m1 | m2)
        y[m2] = 1
        y[m3] = 1 / (1 + np.exp(-x[m3]))
        return y

    @classmethod
    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, y * (1 - y) * dy


class Sum(Function):
    '''
    np.sum along :data:`axis`.

    Args:
        axis (int): the axis along which to sum over.

    Attributes:
        axis (int): the axis along which to sum over.
    '''
    __display_attrs__ = ['axis']

    def __init__(self, input_shape, itype, axis, **kwargs):
        if axis > len(input_shape) - 1:
            raise ValueError('invalid axis')
        self.axis = axis % len(input_shape)
        output_shape = input_shape[:self.axis] + input_shape[self.axis + 1:]
        super(Sum, self).__init__(input_shape, output_shape, itype)

    def forward(self, x, **kwargs):
        return np.sum(x, axis=self.axis)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        if np.ndim(dy) == 0:
            dy_ = np.asarray(dy, order='F')[np.newaxis]
        else:
            dy_ = np.asarray(dy, order='F')[
                (slice(None),) * self.axis + (np.newaxis,)]
        dx = np.repeat(dy_, x.shape[self.axis], axis=self.axis)
        return EMPTY_VAR, dx


class Mean(Function):
    '''
    np.mean along :data:`axis`.

    Args:
        axis (int): the axis along which to operate.

    Attributes:
        axis (int): the axis along which to operate.
    '''
    __display_attrs__ = ['axis']

    def __init__(self, input_shape, itype, axis, **kwargs):
        if axis > len(input_shape) - 1:
            raise ValueError('invalid axis')
        self.axis = axis % len(input_shape)
        output_shape = input_shape[:self.axis] + input_shape[self.axis + 1:]
        super(Mean, self).__init__(input_shape, output_shape, itype)

    def forward(self, x, **kwargs):
        return np.mean(x, axis=self.axis)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        if np.ndim(dy) == 0:
            dy_ = np.asarray(dy, order='F')[np.newaxis]
        else:
            dy_ = np.asarray(dy, order='F')[
                (slice(None),) * self.axis + (np.newaxis,)]
        dx = np.repeat(dy_, x.shape[self.axis],
                       axis=self.axis) / x.shape[self.axis]
        return EMPTY_VAR, dx


class FFT(Function):
    '''
    scipy.fftpack.[fft|ifft|dct|idct|...] along :data:`axis`.

    Args:
        axis (int): the axis along which to operate.
        kernel (str, default='fft'): the kernel used.

    Attributes:
        axis (int): the axis along which to operate.
        kernel (str, default='fft'): the kernel used.
            refer `scipy.fftpack` for available kernels.
    '''
    __display_attrs__ = ['kernel', 'axis']

    def __init__(self, input_shape, itype, axis, kernel='fft', **kwargs):
        if not hasattr(fftpack, kernel):
            raise ValueError(
                'FFT Kernel %s not found in scipy.fftpack!' % kernel)
        dtype = (itype if itype[: 7] == 'complex' else dtype_r2c(itype))\
            if kernel in ['fft', 'ifft', 'fftn', 'ifftn',
                          'fft2', 'ifft2'] else itype
        otype = np.find_common_type((dtype, itype), ())
        super(FFT, self).__init__(input_shape,
                                  input_shape, itype, dtype=dtype, otype=otype)

        self.axis = axis
        self.kernel = kernel

    def forward(self, x, **kwargs):
        func = eval('fftpack.%s' % self.kernel)
        if hasattr(self.axis, '__iter__'):
            return func(x, axes=self.axis)
        else:
            return func(x, axis=self.axis)

    def backward(self, xy, dy, **kwargs):
        # note that dft matrix is hermitian
        func = eval('fftpack.%s' % self.kernel)
        if hasattr(self.axis, '__iter__'):
            dx = func(dy, axes=self.axis)
        else:
            dx = func(dy, axis=self.axis)
        return EMPTY_VAR, dx


class ReLU(Function):
    '''
    ReLU, for mode='ri',

    .. math::
        :nowrap:

        \\begin{align}
        f(x)=\\text{relu}(x)=\\begin{cases}
                x, &\Re[x]>0\land\Im[x]>0\\\\
                \Re[x]+\\text{leak}\cdot\Im[x],&\Re[x]>0\land\Im[x]<0\\\\
                \Im[x]+\\text{leak}\cdot\Re[x],&\Re[x]<0\land\Im[x]>0\\\\
                \\text{leak}\cdot x,&\Re[x]<0\land\Im[x]<0
                \end{cases}
        \\end{align}

    for mode='r',

    .. math::
        :nowrap:

        \\begin{align}
        f(x)=\\text{relu}(x)=\\begin{cases}
                x, &\Re[x]>0\\\\
                \\text{leak}\cdot x,&\Re[x]<0
                \end{cases}
        \\end{align}

    Args:
        leak (float, default = 0.0): leakage,
        mode ('ri'|'r', default='r' if itype is float else 'ri'):
            non-holomophic real-imaginary (ri) relu or holomophic real (r) relu

    Attributes:
        leak (float): leakage,
        mode ('ri'|'r'): non-holomophic real-imaginary (ri)
            relu or holomophic real (r) relu.
    '''
    __display_attrs__ = ['leak']

    def __init__(self, input_shape, itype, leak=0.0, is_inplace=False,
                 mode=None, **kwargs):
        if leak > 1 or leak < 0:
            raise ValueError('leak parameter should be 0-1!')
        self.leak = leak
        if mode is None:
            mode = 'ri' if itype[: 7] == 'complex' else 'r'
        self.mode = mode
        super(ReLU, self).__init__(input_shape, input_shape, itype, tags=dict(
            is_inplace=is_inplace, analytical=3 if mode == 'ri' else 1))

        # use the correct fortran subroutine.
        dtype_token = dtype2token(
            np.find_common_type((self.itype, self.dtype), ()))

        # use the correct function
        self._fforward = eval('frelu.forward_%s%s' % (mode, dtype_token))
        self._fbackward = eval('frelu.backward_%s%s' % (mode, dtype_token))

    def forward(self, x, **kwargs):
        y = self._fforward(x.ravel(order='F'), self.leak).reshape(
            self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        dx = self._fbackward(x=xy[0].ravel(order='F'), dy=dy.ravel(
            order='F'), leak=self.leak).reshape(self.input_shape, order='F')
        return EMPTY_VAR, dx


class Pooling(Function):
    '''
    Pooling, see `Pooling.mode_list` for different types of kernels.

    Args:
        kernel_shape (tuple): the shape of kernel.
        mode (str): the strategy used for pooling,
        refer `Pooling.mode_list` for available modes.

    Attributes:
        kernel_shape (tuple): the shape of kernel.
        mode (str): the strategy used for pooling.

    Note:
        For input array x, axes are aranged as (num_batch, nfi, img_in_dims), \
stored in 'F' order.
        For output array y, axes are aranged as \
(num_batch, nfo, img_out_dims), stored in 'F' order.

        For complex numbers, what does max pooling looks like?
    '''
    __display_attrs__ = ['mode', 'kernel_shape']
    mode_list = ['max', 'max-abs', 'min', 'min-abs', 'mean']

    def __init__(self, input_shape, itype, kernel_shape, mode, **kwargs):
        self.kernel_shape = kernel_shape
        self.mode = mode
        if mode not in self.mode_list:
            raise ValueError('mode %s not allowed!' % mode)
        img_in_shape = input_shape[-len(kernel_shape):]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(
            kernel_shape, img_in_shape, strides=kernel_shape, boundary='O')
        output_shape = input_shape[: -len(kernel_shape)] + self.img_out_shape
        super(Pooling, self).__init__(input_shape, output_shape, itype)

        # use the correct fortran subroutine.
        dtype_token = dtype2token(
            np.find_common_type((self.itype, self.dtype), ()))

        # use the correct function
        self._fforward = eval('fpooling.forward_%s' % dtype_token)
        self._fbackward = eval('fpooling.backward_%s' % dtype_token)

    @property
    def img_nd(self):
        '''int: dimension of image.'''
        return len(self.kernel_shape)

    def forward(self, x, **kwargs):
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim = tuple_prod(self.input_shape[-img_nd:])

        y = self._fforward(x.reshape([-1, img_dim], order='F'),
                           csc_indptr=self.csc_indptr,
                           csc_indices=self.csc_indices,
                           mode=self.mode_list.index(self.mode)
                           ).reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim_in = tuple_prod(self.input_shape[-img_nd:])
        img_dim_out = tuple_prod(self.output_shape[-img_nd:])

        dx = self._fbackward(x=x.reshape([-1, img_dim_in], order='F'),
                             dy=dy.reshape([-1, img_dim_out], order='F'),
                             csc_indptr=self.csc_indptr,
                             csc_indices=self.csc_indices,
                             mode=self.mode_list.index(self.mode)
                             ).reshape(self.input_shape, order='F')
        return EMPTY_VAR, dx


class ConvProd(Function):
    '''
    Convolutional product layer, apply a kernel as powers to a subregion
    and make product over these elements.

    Args:
        powers (ndarray): powers as a kernel.
        strides (tuple, default=(1, 1,...)): stride for each dimension.
        boundary ('P'|'O', default='O'): Periodic/Open boundary condition.

    Attributes:
        powers (ndarray): powers as a kernel.
        strides (tuple): stride for each dimension.
        boundary ('P'|'O'): Periodic/Open boundary condition.

    Note:
        For input array x, axes are aranged as (num_batch, \
nfi, img_in_dims), stored in 'F' order.
        For output array y, axes are aranged as (num_batch, nfo, \
img_out_dims), stored in 'F' order.
    '''
    __display_attrs__ = ['powers', 'strides', 'boundary']

    def __init__(self, input_shape, itype, powers, strides=None,
                 boundary='O', **kwargs):
        self.boundary = boundary
        self.powers = np.asarray(powers, order='F')

        img_nd = self.powers.ndim
        if strides is None:
            strides = (1,) * img_nd
        self.strides = strides

        img_in_shape = input_shape[-img_nd:]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(
            self.powers.shape, img_in_shape, strides=strides,
            boundary=boundary)
        output_shape = input_shape[: -img_nd] + self.img_out_shape
        super(ConvProd, self).__init__(input_shape, output_shape, itype)

        # use the correct fortran subroutine.
        dtype_token = dtype2token(
            np.find_common_type((self.itype, self.dtype), ()))

        # use the correct function
        self._fforward = eval('fconvprod.forward_%s' % dtype_token)
        self._fbackward = eval('fconvprod.backward_%s' % dtype_token)

    @property
    def img_nd(self):
        '''int: dimension of image.'''
        return self.powers.ndim

    def forward(self, x, **kwargs):
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim = tuple_prod(self.input_shape[-img_nd:])
        y = self._fforward(x.reshape([-1, img_dim], order='F'),
                           csc_indptr=self.csc_indptr,
                           powers=self.powers, csc_indices=self.csc_indices
                           ).reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim_in = tuple_prod(self.input_shape[-img_nd:])
        img_dim_out = tuple_prod(self.output_shape[-img_nd:])

        dx = self._fbackward(x=x.reshape([-1, img_dim_in], order='F'),
                             dy=dy.reshape([-1, img_dim_out], order='F'),
                             y=y.reshape([-1, img_dim_out], order='F'),
                             powers=self.powers, csc_indptr=self.csc_indptr,
                             csc_indices=self.csc_indices).reshape(
            self.input_shape, order='F')
        return EMPTY_VAR, dx


class DropOut(Function):
    '''
    DropOut, take runtime variable seed.

    Args:
        axis (int): the axis along which to operate.
        keep_rate (float): the ratio of kept data.

    Attributes:
        axis (int): the axis along which to operate.
        keep_rate (float): the ratio of kept data.

    Example:
        >>> layer = DropOut((3, 3), 'complex128', keep_rate = 0.5, axis=-1)
        >>> layer.set_runtime_vars({'seed': 2})
        >>> x = np.arange(9, dtype='complex128').reshape([3, 3], order='F')
        >>> print(layer.forward(x))
        [[  0.+0.j   6.+0.j   0.+0.j]
         [  2.+0.j   8.+0.j   0.+0.j]
         [  4.+0.j  10.+0.j   0.+0.j]]
    '''
    __display_attrs__ = ['axis', 'keep_rate']

    def __init__(self, input_shape, itype, keep_rate, axis,
                 is_inplace=False, **kwargs):
        if axis > len(input_shape) - 1:
            raise ValueError('invalid axis')
        self.axis = axis % len(input_shape)
        self.keep_rate = keep_rate
        self.seed = None
        self.mask = None
        super(DropOut, self).__init__(input_shape, input_shape, itype,
                                      tags=dict(runtimes=['seed'],
                                                is_inplace=is_inplace))

    def set_runtime_vars(self, var_dict):
        '''Set the runtime variable seed, used to generate a random mask.'''
        super(DropOut, self).set_runtime_vars(var_dict)
        np.random.seed(self.seed)
        self.mask = np.random.random(
            self.input_shape[self.axis]) < self.keep_rate

    def forward(self, x, **kwargs):
        if self.seed is None:
            raise AttributeError('Please initialize variable\
                                 seed(use @set_runtime_vars)\
                                 before using a runtime layer % s!' % self)
        y = x if self.tags['is_inplace'] else x.copy(order='F')
        y[(slice(None),) * self.axis + (self.mask,)] /= self.keep_rate
        y[(slice(None),) * self.axis + (~self.mask,)] = 0
        return y

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        dy[(slice(None),) * self.axis + (self.mask,)] /= self.keep_rate
        dy[(slice(None),) * self.axis + (~self.mask,)] = 0
        return EMPTY_VAR, dy


class SoftMax(Function):
    '''
    Soft max function
    :math:`f(x)=\\text{scale}\cdot\\frac{\exp(x)}{\sum \exp(x)}`,
    with the sum performed over :data:`axis`.

    Args:
        axis (int): the axis along which to operate.
        scale (number): the factor to rescale output.

    Attributes:
        axis (int): the axis along which to operate.
        scale (number, default = 1.0): the factor to rescale output.
    '''
    __display_attrs__ = ['axis', 'scale']

    def __init__(self, input_shape, itype, axis, scale=1., **kwargs):
        self.axis = axis
        self.scale = scale
        super(SoftMax, self).__init__(input_shape, input_shape, itype)

    def forward(self, x, **kwargs):
        x = x - x.max(axis=self.axis, keepdims=True)
        rho = np.exp(x)
        return self.scale * rho / rho.sum(axis=self.axis, keepdims=True)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, (dy * y - (dy * y).sum(axis=self.axis,
                                                 keepdims=True) *
                           y / self.scale)


class CrossEntropy(Function):
    '''
    Cross Entropy :math:`f(x)=\sum\\text{y_true}\log(x)`,
        with y_true the true labels,
        and the sum is performed over :data:`axis`.

    Args:
        axis (int): the axis along which to operate.

    Attributes:
        axis (int): the axis along which to operate.
    '''
    ZERO_REF = 1e-15
    __display_attrs__ = ['axis']

    def __init__(self, input_shape, itype, axis, **kwargs):
        if axis > len(input_shape) - 1:
            raise ValueError('invalid axis')
        self.axis = axis % len(input_shape)
        self.y_true = None
        output_shape = input_shape[: self.axis] + input_shape[self.axis + 1:]
        super(CrossEntropy, self).__init__(input_shape,
                                           output_shape, itype,
                                           tags=dict(runtimes=['y_true']))

    def forward(self, x, **kwargs):
        '''
        Args:
            x (ndarray): satisfying :math:`0 < x \leq 1`.
            y_true (ndarray): correct one-hot y.
        '''
        return (-self.y_true * scipy.log(np.maximum(self.ZERO_REF, x))
                ).sum(axis=self.axis)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, -dy[(slice(None),) * self.axis + (np.newaxis,)]\
            * (self.y_true / np.maximum(x, self.ZERO_REF))


class SoftMaxCrossEntropy(Function):
    '''
    Soft Max & Cross Entropy :math:`f(x)=\sum \\text{y_true}\log(q)`,
    with y_true the true labels, and :math:`q=\\frac{\exp(x)}{\sum \exp(x)}`,
    where the sum is performed over :data:`axis`.

    Args:
        axis (int): the axis along which to operate.

    Attributes:
        axis (int): the axis along which to operate.
    '''
    __display_attrs__ = ['axis']

    def __init__(self, input_shape, itype, axis, **kwargs):
        if axis > len(input_shape) - 1:
            raise ValueError('invalid axis')
        self.axis = axis % len(input_shape)
        self.y_true = None
        output_shape = input_shape[: axis] + input_shape[self.axis + 1:]
        super(SoftMaxCrossEntropy, self).__init__(input_shape,
                                                  output_shape,
                                                  itype,
                                                  tags=dict(
                                                      runtimes=['y_true']))

    def forward(self, x, **kwargs):
        if self.y_true is None:
            raise AttributeError('Please initialize variable \
y_true(use @set_runtime_vars) \
before using a runtime layer % s!' % self)
        x = x - x.max(axis=self.axis, keepdims=True)
        rho = np.exp(x)
        Z = rho.sum(axis=self.axis, keepdims=True)
        return ((scipy.log(Z) - x) * self.y_true).sum(axis=self.axis)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        x = x - x.max(axis=self.axis, keepdims=True)
        rho = np.exp(x)
        Z = rho.sum(axis=self.axis, keepdims=True)
        y1 = rho / Z
        return EMPTY_VAR, dy[(slice(None),) * self.axis + (np.newaxis,)] *\
            (y1 - self.y_true)


class SquareLoss(Function):
    '''
    Square Loss :math:`f(x)=(x-\\text{y_true})^2`. With p the true labels, \
requres runtime variable 'y_true'.

    Attributes:
        y_true (ndarray): the 'correct' output.
    '''

    def __init__(self, input_shape, itype, **kwargs):
        self.y_true = None
        super(SquareLoss, self).__init__(input_shape, input_shape,
                                         itype, tags=dict(runtimes=['y_true'],
                                                          analytical=2))

    def forward(self, x, **kwargs):
        if self.y_true is None:
            raise AttributeError('Please initialize variable \
y_true(use @set_runtime_vars) before using a runtime layer % s!' % self)
        diff = x - self.y_true
        return (diff.conj() * diff).real

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        xt = self.y_true
        is_complex = self.itype[: 7] == 'complex'
        return EMPTY_VAR, ((x - xt).conj() * dy.real * 2)\
            if is_complex else (2 * (xy[0] - xt) * dy)


class Reshape(Function):
    '''
    Change shape of data, reshape is performed in 'F' order.

    Note:
        output_shape is a mandatory parameter now.
    '''

    def forward(self, x, **kwargs):
        return x.reshape(self.output_shape, order='F')

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, dy.reshape(self.input_shape, order='F')

class Reverse(Function):
    '''
    Reverse data along specific axis.
    '''
    def __init__(self, input_shape, itype, axis, **kwargs):
        if axis > len(input_shape) - 1:
            raise ValueError('invalid axis')
        self.axis = axis % len(input_shape)
        super(Reverse, self).__init__(
            input_shape, input_shape, itype, otype=itype)

    def forward(self, x, **kwargs):
        return x[(slice(None),)*self.axis+(slice(None,None,-1),)]

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, dy[(slice(None),)*self.axis+(slice(None,None,-1),)]


class TypeCast(Function):
    '''
    Data type switcher.

    Note:
        otype is a mandatory parameter now.
    '''

    def __init__(self, input_shape, itype, otype, **kwargs):
        super(TypeCast, self).__init__(
            input_shape, input_shape, itype, otype=otype)

    def forward(self, x, **kwargs):
        return np.asarray(x, dtype=self.otype, order='F')

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, np.asarray(dy, dtype=self.itype, order='F')


class Transpose(Function):
    '''
    Transpose data flow.

    Args:
        axes (tuple): the target axes order.

    Attributes:
        axes (tuple): the target axes order.
    '''
    __display_attrs__ = ['axes']

    def __init__(self, input_shape, itype, axes, **kwargs):
        self.axes = axes
        if len(axes) != len(input_shape):
            raise ValueError('axes incorrect!')
        output_shape = tuple([input_shape[axis] for axis in self.axes])
        super(Transpose, self).__init__(input_shape, output_shape, itype)

    def forward(self, x, **kwargs):
        return x.transpose(self.axes)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, dy.transpose(np.argsort(self.axes))


class Filter(Function):
    '''
    Momentum Filter, single component fourier transformation.
    :math:`f(x)=\sum\limits_{n = 0}^{N-1}\exp(-i\pi k n/N)\cdot x[n]`, \
with index :math:`n` iterate over :data:`axes`.

    Args:
        momentum (1darray): the desired momentum.
        axes (tuple): lattice axes over which to filter \
out component with the desired momentum.

    Attributes:
        momentum (1darray): the desired momentum.
        axes (tuple): lattice axes over which to filter \
out component with the desired momentum.
    '''
    __display_attrs__ = ['momentum', 'axes']

    def __init__(self, input_shape, itype, momentum, axes, **kwargs):
        # sort axes and momentum
        DIM = len(input_shape)
        axes = [axis % DIM for axis in axes]
        order = np.argsort(axes)
        axes = tuple([axes[od] for od in order])
        momentum = np.atleast_1d(momentum)[order]

        self.momentum = momentum
        self.axes = axes
        size = [input_shape[axis] for axis in axes]
        self.filters = [np.exp(-1j * momentum *
                               np.arange(ni)
                               ).reshape([1] * axis +
                                         [-1] + [1] * (DIM - axis - 1)) / ni
                        for ki, axis, ni in zip(momentum, axes, size)]
        if all(momentum % np.pi == 0):
            self.filters = [flt.real.astype(itype) for flt in self.filters]
        else:
            self.filters = [flt.astype(itype) for flt in self.filters]
        output_shape = tuple(
            [dim for axis, dim in enumerate(input_shape) if axis not in axes])
        # np.prod(np.ix_([np.exp(-1j*k*np.arange(ni))/ni for ki, ni
        # in zip(momentum, size)]), axis = 0)
        super(Filter, self).__init__(input_shape, output_shape, itype)

    def forward(self, x, **kwargs):
        y = x
        for axis, fltr in zip(self.axes[:: -1], self.filters[:: -1]):
            y = (y * fltr).sum(axis=axis)
        return y

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        dx = dy
        for axis, fltr in zip(self.axes, self.filters):
            dx = np.asarray(dx, order='F')[
                (slice(None),) * axis + (np.newaxis,)] *\
                fltr.reshape(fltr.shape[: dx.ndim + 1])
        return EMPTY_VAR, dx


class BatchNorm(Function):
    '''
    Batch normalization layer.

    Args:
        axis (int|None, default = None): batch axis over which we take norm.
        eps (float, default = 1e-8): small number to avoid division to 0.

    Attributes:
        axis (int|None): batch axis over which to calculate norm, \
if it is None, we don't use any axis as batch, \
instead, we need to set mean and variance manually.
        eps (float): small number to avoid division to 0.

    Note:
        shall we take mean and variance as run time variable?
    '''
    __display_attrs__ = ['axis']

    def __init__(self, input_shape, itype, eps=1e-8, axis=None, **kwargs):
        super(BatchNorm, self).__init__(input_shape, input_shape, itype)
        self.mean = None
        self.variance = None
        self.eps = eps

        if axis is not None:
            if axis > len(input_shape) - 1:
                raise ValueError('invalid axis')
            axis = axis % len(input_shape)
        self.axis = axis

    def forward(self, x, **kwargs):
        if self.axis is not None:
            self.mean = x.mean(axis=self.axis, keepdims=True)
            self.variance = np.var(x, axis=self.axis, keepdims=True)
        elif self.mean is None or self.variance is None:
            raise Exception('mean and variance not initialized!')
        return (x - self.mean) / np.sqrt(self.variance + self.eps)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, dy / np.sqrt(self.variance + self.eps)


class Normalize(Function):
    '''
    Normalize data, :math:`f(x)=\\text{scale}\cdot x/\\|x\\|`, \
where the norm is performed over :data:`axis`.

    Args:
        axis (int): axis over which to calculate norm.
        scale (number, default = 1.0): the scaling factor.

    Attributes:
        axis (int): axis over which to calculate norm.
        scale (number): the scaling factor.
    '''
    __display_attrs__ = ['axis', 'scale']

    def __init__(self, input_shape, itype, axis, scale=1., **kwargs):
        super(Normalize, self).__init__(input_shape,
                                        input_shape, itype,
                                        tags={'analytical': 3})

        self.scale = scale
        if axis is not None:
            if axis > len(input_shape) - 1:
                raise ValueError('invalid axis')
            axis = axis % len(input_shape)
        self.axis = axis

    def forward(self, x, **kwargs):
        return self.scale * x / np.linalg.norm(x,
                                               axis=self.axis, keepdims=True)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        norm = np.linalg.norm(x, axis=self.axis, keepdims=True)
        cum = (x * dy).sum(axis=self.axis, keepdims=True).real
        return EMPTY_VAR, (dy / norm - x.conj() * cum / norm**3) * self.scale


Sin = wrapfunc(np.sin, lambda xy, dy: np.cos(xy[0]) * dy,
               classname='Sin',
               docstring="Function :math:`f(x)=\sin(x)`")
Sinh = wrapfunc(np.sinh, lambda xy, dy: np.cosh(xy[0]) * dy,
                classname='Sinh',
                docstring="Function :math:`f(x)=\sinh(x)`")
Cos = wrapfunc(np.cos, lambda xy, dy: -np.sin(xy[0]) * dy,
               classname='Cos',
               docstring="Function :math:`f(x)=\cos(x)`")
Cosh = wrapfunc(np.cosh, lambda xy, dy: np.sinh(xy[0]) * dy,
                classname='Cosh',
                docstring="Function :math:`f(x)=\\cosh(x)`")
Tan = wrapfunc(np.tan, lambda xy, dy: 1. / np.cos(xy)[0]**2 * dy,
               classname='Tan',
               docstring="Function :math:`f(x)=\\tan(x)`")
Tanh = wrapfunc(np.tanh, lambda xy, dy: 1. / np.cosh(xy)[0]**2 * dy,
                classname='Tanh',
                docstring="Function :math:`f(x)=\\tanh(x)`")
ArcTan = wrapfunc(np.arctan, lambda xy, dy: 1. / (1 + xy[0]**2) * dy,
                  classname='ArcTan',
                  docstring="Function :math:`f(x)=\\arctan(x)`")

Exp = wrapfunc(np.exp, lambda xy, dy: xy[1] * dy,
               classname='Exp',
               docstring="Function :math:`f(x)=\exp(x)`")
Log = wrapfunc(scipy.log, lambda xy, dy: dy / xy[0],
               classname='Log',
               docstring="Function :math:`f(x)=\log(x)`")
SoftPlus = wrapfunc(lambda x: scipy.log(1 + np.exp(x)),
                    lambda xy, dy: dy * Sigmoid.forward(xy[0]),
                    classname='SoftPlus',
                    docstring="Function :math:`log(1+exp(x))`")

Conj = wrapfunc(np.conj, lambda xy, dy: dy.conj(),
                classname='Conj',
                docstring="Function :math:`f(x)=x^*`",
                tags={'analytical': 3})
Real = wrapfunc(np.real, lambda xy, dy: dy.real,
                classname='Real',
                docstring="Function :math:`f(x)=\Re[x]`",
                tags={'analytical': 2}, real_out=True)
Imag = wrapfunc(np.imag, lambda xy, dy: -1j * dy.real,
                classname='Imag',
                docstring="Function :math:`f(x)=\Im[x]`",
                tags={'analytical': 2}, real_out=True)
Abs = wrapfunc(np.abs, lambda xy, dy: xy[0].conj() / np.abs(xy[0]) * dy.real,
               classname='Abs',
               docstring="Function :math:`f(x)=|x|`",
               tags={'analytical': 2}, real_out=True)
Abs2 = wrapfunc(lambda x: np.abs(x)**2,
                lambda xy, dy: 2 * xy[0].conj() * dy.real,
                classname='Abs2',
                docstring="Function :math:`f(x)=|x|^2`",
                tags={'analytical': 2}, real_out=True)
Sign = wrapfunc(lambda x: fsign,
                lambda xy, dy: xy[1].conj() / np.abs(x) * 1j * (y * dy).imag,
                classname='Sign',
                docstring="Function ::math:`f(x)=x/|x|`",
                tags={'analytical': 3})
Angle = wrapfunc(lambda x: np.angle(x), lambda xy, dy: -1j / xy[0] * dy.real,
                 classname='Angle',
                 docstring="Function :math:`f(x)=\\text{Arg}(x)`",
                 tags={'analytical': 2}, real_out=True)

Mul = wrapfunc(lambda x, alpha: x * alpha, lambda xy, dy, alpha: alpha * dy,
               attrs={'alpha': None}, classname='Mul', docstring='''
        Function :math:`f(x)=\\text{alpha}\cdot x`

        Args:
            alpha (int): the multiplier.

        Attributes:
            alpha (int): the multiplier.
        ''')
Mod = wrapfunc(lambda x, n: x % n, lambda xy, dy, n: dy, attrs={'n': None},
               classname='Mod', docstring='''
        Function :math:`f(x)=x\%n`

        Args:
            n (number): the base.

        Attributes:
            n (number): the base.
        ''')
Power = wrapfunc(lambda x, order: x**order,
                 lambda xy, dy, order: order * xy[0]**(order - 1) * dy,
                 attrs={'order': None},
                 classname='Power', docstring='''
        Function :math:`f(x)=x^{\\rm order}`

        Args:
            order (number): the order of power.

        Attributes:
            order (number): the order of power.
        ''')
