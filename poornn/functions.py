import numpy as np
import scipy
from numbers import Number
import pdb

from .core import Layer, Function, EXP_OVERFLOW, EMPTY_VAR
from .lib.pooling import lib as fpooling
from .lib.convprod import lib as fconvprod
from .lib.relu import lib as frelu
from .utils import scan2csc, tuple_prod, dtype2token

__all__=['Log2cosh','Sigmoid','Cosh','Sinh','Tanh','Sum','Mul','Mod','Mean','ReLU','ConvProd',
        'Pooling','DropOut','Sin','Cos','Exp','Log','SoftPlus','Power',
        'SoftMax','CrossEntropy','SoftMaxCrossEntropy','SquareLoss', 'Reshape','Transpose',
        'TypeCast', 'Print', 'Cache', 'Filter']

class Log2cosh(Function):
    '''
    Function log(2*cosh(theta)).
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Log2cosh, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        if np.ndim(x)==0:
            return scipy.log(2*np.cosh(x)) if abs(x.real)<=12 else np.sign(x.real)*x
        x=np.asarray(x)
        res=np.zeros_like(x)
        m1=x.real>EXP_OVERFLOW
        m2=x.real<-EXP_OVERFLOW
        m3=~(m1|m2)
        res[m1]=x[m1]
        res[m2]=-x[m2]
        res[m3]=scipy.log(2*np.cosh(x[m3]))
        return res

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,np.tanh(x)*dy


class Cosh(Function):
    '''
    Function cosh(theta)
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Cosh, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        return np.cosh(x)

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,np.sinh(x)*dy

class Sinh(Function):
    '''
    Function sinh(theta)
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Sinh, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        return np.sinh(x)

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,np.cosh(x)*dy

class Tanh(Function):
    '''
    Function tanh(theta)
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Tanh, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        return np.tanh(x)

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,1./np.cosh(x)**2*dy

class Sigmoid(Function):
    '''
    Function 1/(1+exp(-x))
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Sigmoid, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        #for ndarray
        y=np.zeros_like(x)
        m1=x.real<-EXP_OVERFLOW
        m2=x.real>EXP_OVERFLOW
        m3=~(m1|m2)
        y[m2]=1
        y[m3]=1/(1+np.exp(-x[m3]))
        return y

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,y*(1-y)*dy

class Sum(Function):
    '''
    Sum along specific axis.
    '''
    __display_attrs__ = ['axis']
    def __init__(self, input_shape, itype, axis, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        super(Sum,self).__init__(input_shape, output_shape, itype)

    def forward(self,x):
        return np.sum(x,axis=self.axis)
    
    def backward(self, xy, dy, **kwargs):
        x, y = xy
        if np.ndim(dy)==0:
            dy_ = np.asarray(dy, order='F')[np.newaxis]
        else:
            dy_ = np.asarray(dy, order='F')[(slice(None),)*self.axis+(np.newaxis,)]
        dx = np.repeat(dy_,x.shape[self.axis],axis=self.axis)
        return EMPTY_VAR, dx

class Mean(Function):
    '''
    Mean along specific axis.
    '''
    __display_attrs__ = ['axis']
    def __init__(self,input_shape, itype, axis, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        super(Mean,self).__init__(input_shape, output_shape, itype)

    def forward(self,x):
        return np.mean(x,axis=self.axis)
    
    def backward(self, xy, dy, **kwargs):
        x, y = xy
        if np.ndim(dy)==0:
            dy_ = np.asarray(dy, order='F')[np.newaxis]
        else:
            dy_ = np.asarray(dy, order='F')[(slice(None),)*self.axis+(np.newaxis,)]
        dx = np.repeat(dy_,x.shape[self.axis],axis=self.axis)/x.shape[self.axis]
        return EMPTY_VAR, dx


class ReLU(Function):
    '''
    ReLU.
    '''
    __display_attrs__ = ['leak']
    def __init__(self, input_shape, itype, leak = 0, is_inplace=False, **kwargs):
        super(ReLU,self).__init__(input_shape, input_shape, itype, tags=dict(is_inplace=is_inplace))
        if leak>1 or leak<0:
            raise ValueError('leak parameter should be 0-1!')
        self.leak = leak

        #use the correct fortran subroutine.
        dtype_token = dtype2token(np.find_common_type((self.itype,self.dtype),()))

        #use the correct function
        self._fforward=eval('frelu.forward_%s'%dtype_token)
        self._fbackward=eval('frelu.backward_%s'%dtype_token)

    def forward(self, x):
        y=self._fforward(x.ravel(order='F'),self.leak).reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        dx=self._fbackward(x=xy[0].ravel(order='F'),dy=dy.ravel(order='F'),leak=self.leak).reshape(self.input_shape, order='F')
        return EMPTY_VAR, dx

class Pooling(Function):
    '''
    Max/Mean pooling.

    Note:
        for complex numbers, what does max pooling looks like?
    '''
    __display_attrs__ = ['mode', 'kernel_shape']
    mode_list = ['max', 'max-abs', 'min', 'min-abs', 'mean']

    def __init__(self, input_shape, itype, kernel_shape, mode, **kwargs):
        self.kernel_shape = kernel_shape
        self.mode = mode
        if mode not in self.mode_list:
            raise ValueError('mode %s not allowed!'%mode)
        img_in_shape = input_shape[-len(kernel_shape):]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(kernel_shape, img_in_shape, strides=kernel_shape, boundary='O')
        output_shape = input_shape[:-len(kernel_shape)]+self.img_out_shape
        super(Pooling,self).__init__(input_shape, output_shape, itype)

        #use the correct fortran subroutine.
        dtype_token = dtype2token(np.find_common_type((self.itype,self.dtype),()))

        #use the correct function
        self._fforward=eval('fpooling.forward_%s'%dtype_token)
        self._fbackward=eval('fpooling.backward_%s'%dtype_token)

    @property
    def img_nd(self):
        return len(self.kernel_shape)

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), input in 'F' order.

        Return:
            ndarray, (num_batch, nfo, img_out_dims), output in 'F' order.
        '''
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim = tuple_prod(self.input_shape[-img_nd:])

        y=self._fforward(x.reshape([-1,img_dim], order='F'), csc_indptr=self.csc_indptr,\
                csc_indices=self.csc_indices, mode=self.mode_list.index(self.mode)).reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        '''It will shed a mask on dy'''
        x, y = xy
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim_in = tuple_prod(self.input_shape[-img_nd:])
        img_dim_out = tuple_prod(self.output_shape[-img_nd:])

        dx=self._fbackward(x=x.reshape([-1,img_dim_in], order='F'), dy=dy.reshape([-1,img_dim_out], order='F'),\
                csc_indptr=self.csc_indptr,csc_indices=self.csc_indices, mode=self.mode_list.index(self.mode)).reshape(self.input_shape, order='F')
        return EMPTY_VAR, dx

class ConvProd(Function):
    '''
    Convolutional product layer.
    '''
    __display_attrs__ = ['powers', 'strides', 'boundary']
    def __init__(self, input_shape, itype, powers, strides=None, boundary='O', **kwargs):
        self.boundary = boundary
        self.powers = np.asarray(powers, order='F')

        img_nd = self.powers.ndim
        if strides is None:
            strides=(1,)*img_nd
        self.strides = strides

        img_in_shape = input_shape[-img_nd:]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(self.powers.shape, img_in_shape, strides=strides, boundary=boundary)
        output_shape = input_shape[:-img_nd]+self.img_out_shape
        super(ConvProd,self).__init__(input_shape, output_shape, itype)

        #use the correct fortran subroutine.
        dtype_token = dtype2token(np.find_common_type((self.itype,self.dtype),()))

        #use the correct function
        self._fforward=eval('fconvprod.forward_%s'%dtype_token)
        self._fbackward=eval('fconvprod.backward_%s'%dtype_token)

    @property
    def img_nd(self):
        return self.powers.ndim

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), input in 'F' order.

        Return:
            ndarray, (num_batch, nfo, img_out_dims), output in 'F' order.
        '''
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim = tuple_prod(self.input_shape[-img_nd:])
        y=self._fforward(x.reshape([-1,img_dim], order='F'), csc_indptr=self.csc_indptr,\
                powers=self.powers, csc_indices=self.csc_indices).reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        '''It will shed a mask on dy'''
        x, y = xy
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim_in = tuple_prod(self.input_shape[-img_nd:])
        img_dim_out = tuple_prod(self.output_shape[-img_nd:])


        dx=self._fbackward(x=x.reshape([-1,img_dim_in], order='F'), dy=dy.reshape([-1,img_dim_out], order='F'), y=y.reshape([-1,img_dim_out], order='F'),\
                powers=self.powers, csc_indptr=self.csc_indptr,csc_indices=self.csc_indices).reshape(self.input_shape, order='F')
        return EMPTY_VAR, dx

class DropOut(Function):
    '''
    DropOut inplace.
    '''
    __display_attrs__ = ['axis', 'keep_rate']

    def __init__(self, input_shape, itype, keep_rate, axis, is_inplace=False, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        self.keep_rate = keep_rate
        self.seed = None
        self.mask = None
        super(DropOut, self).__init__(input_shape, input_shape, itype, tags=dict(runtimes=['seed'],is_inplace=is_inplace))

    def set_runtime_vars(self, var_dict):
        '''Set the runtime variable by seed.'''
        super(DropOut, self).set_runtime_vars(var_dict)
        np.random.seed(self.seed)
        self.mask = np.random.random(self.input_shape[self.axis])<self.keep_rate

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (num_batch, num_feature_in), in fortran order.
        '''
        if self.seed is None:
            raise AttributeError('Please initialize variable `seed`(use @set_runtime_vars) before using a runtime layer %s!'%self)
        y=x if self.tags['is_inplace'] else x.copy(order='F')
        y[(slice(None),)*self.axis+(self.mask,)]/=self.keep_rate
        y[(slice(None),)*self.axis+(~self.mask,)]=0
        return y

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        dy[(slice(None),)*self.axis+(self.mask,)]/=self.keep_rate
        dy[(slice(None),)*self.axis+(~self.mask,)]=0
        return EMPTY_VAR, dy

class SoftMax(Function):
    '''
    Soft max function applied on the last axis.
    '''
    __display_attrs__ = ['axis']

    def __init__(self, input_shape, itype, axis, **kwargs):
        self.axis=axis
        super(SoftMax, self).__init__(input_shape, input_shape, itype)

    def forward(self, x):
        x=x-x.max(axis=self.axis, keepdims=True)
        rho=np.exp(x)
        return rho/rho.sum(axis=self.axis, keepdims=True)

    def backward(self, xy, dy, **kwargs):
        x,y = xy
        return EMPTY_VAR,dy*y-(dy*y).sum(axis=self.axis, keepdims=True)*y

class CrossEntropy(Function):
    '''
    Cross Entropy sum(p*log(q)). With p the true labels.
        q = x
    '''
    ZERO_REF=1e-15
    __display_attrs__ = ['axis']

    def __init__(self, input_shape, itype, axis, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        self.y_true = None
        output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        super(CrossEntropy, self).__init__(input_shape, output_shape, itype, tags=dict(runtimes=['y_true']))

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, note 0 < x <= 1.
            :y_true: ndarray, correct one-hot y.
        '''
        return (-self.y_true*scipy.log(np.maximum(self.ZERO_REF,x))).sum(axis=self.axis)

    def backward(self, xy, dy, **kwargs):
        x,y = xy
        return EMPTY_VAR,-dy[(slice(None),)*self.axis+(np.newaxis,)]*(self.y_true/np.maximum(x, self.ZERO_REF))

class SoftMaxCrossEntropy(Function):
    '''
    Cross Entropy sum(p*log(q)). With p the true labels.
        q = exp(x)/sum(exp(x))
    '''
    __display_attrs__ = ['axis']

    def __init__(self, input_shape, itype, axis, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        self.y_true = None
        output_shape = input_shape[:axis]+input_shape[self.axis+1:]
        super(SoftMaxCrossEntropy, self).__init__(input_shape, output_shape, itype, tags=dict(runtimes=['y_true']))

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, note 0 < x <= 1.
            :y_true: ndarray, correct one-hot y.
        '''
        if self.y_true is None:
            raise AttributeError('Please initialize variable `y_true`(use @set_runtime_vars) before using a runtime layer %s!'%self)
        x=x-x.max(axis=self.axis, keepdims=True)
        rho=np.exp(x)
        Z=rho.sum(axis=self.axis, keepdims=True)
        return ((scipy.log(Z)-x)*self.y_true).sum(axis=self.axis)

    def backward(self, xy, dy, **kwargs):
        x,y = xy
        x=x-x.max(axis=self.axis, keepdims=True)
        rho=np.exp(x)
        Z=rho.sum(axis=self.axis, keepdims=True)
        y1=rho/Z
        return EMPTY_VAR,dy[(slice(None),)*self.axis+(np.newaxis,)]*(y1-self.y_true)

class SquareLoss(Function):
    '''
    Square Loss (p-q)**2. With p the true labels.
    '''
    def __init__(self, input_shape, itype, **kwargs):
        self.y_true = None
        super(SquareLoss, self).__init__(input_shape, input_shape, itype, tags=dict(runtimes=['y_true'],analytical=2))

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, note 0 < x <= 1.
            :y_true: ndarray, correct one-hot y.
        '''
        if self.y_true is None:
            raise AttributeError('Please initialize variable `y_true`(use @set_runtime_vars) before using a runtime layer %s!'%self)
        diff=x-self.y_true
        return diff.conj()*diff

    def backward(self, xy, dy, **kwargs):
        xt=self.y_true
        is_complex = self.itype[:7]=='complex'
        #return EMPTY_VAR,((xy[0]-xt).conj()*dy) if self.itype[:7]=='complex' else (2*(xy[0]-xt)*dy)
        return EMPTY_VAR,((xy[0]-xt)*dy) if is_complex else (2*(xy[0]-xt)*dy)  #paritial x.conj() for complex

class Exp(Function):
    '''
    Function exp(x)
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Exp, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        return np.exp(x)

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,dy*y

class Log(Function):
    '''
    Function log(x)
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Log, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        return scipy.log(x)

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,dy/x


class Sin(Function):
    '''
    Function sin(x)
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Sin, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        return np.sin(x)

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,dy*np.cos(x)

class Cos(Function):
    '''
    Function cos(x)
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Cos, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        return np.cos(x)

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,-dy*np.sin(x)

class SoftPlus(Function):
    '''
    Function log(1+exp(x))
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(SoftPlus, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self,x):
        return scipy.log(1+np.exp(x))

    @classmethod
    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR,dy*Sigmoid.forward(x)

class Reshape(Function):
    def forward(self, x):
        return x.reshape(self.output_shape, order='F')

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, dy.reshape(self.input_shape, order='F')

class TypeCast(Function):
    def __init__(self, input_shape, itype, otype, **kwargs):
        super(TypeCast, self).__init__(input_shape, input_shape, itype, otype=otype)

    def forward(self, x):
        return np.asarray(x, dtype=self.otype, order='F')

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, np.asarray(dy, dtype=self.itype, order='F')

class Transpose(Function):
    __display_attrs__ = ['axes']
    def __init__(self, input_shape, itype, axes, **kwargs):
        self.axes=axes
        if len(axes)!=len(input_shape):
            raise ValueError('axes incorrect!')
        output_shape=tuple([input_shape[axis] for axis in self.axes])
        super(Transpose, self).__init__(input_shape, output_shape, itype)

    def forward(self, x):
        return x.transpose(self.axes)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, dy.transpose(np.argsort(self.axes))

class Mul(Function):
    '''Multiply by a constant'''
    __display_attrs__ = ['alpha']

    def __init__(self, input_shape, itype, alpha, **kwargs):
        self.alpha = alpha
        super(Mul, self).__init__(input_shape, input_shape, itype)

    def forward(self, x):
        return self.alpha*x

    def backward(self,xy, dy, **kwargs):
        return EMPTY_VAR, self.alpha*dy

class Mod(Function):
    '''Mod by a constant'''
    __display_attrs__ = ['n']

    def __init__(self, input_shape, itype, n, **kwargs):
        self.n = n
        super(Mod, self).__init__(input_shape, input_shape, itype)

    def forward(self, x):
        return x%self.n

    @classmethod
    def backward(self,xy, dy, **kwargs):
        return EMPTY_VAR, dy

class Print(Function):
    '''Print data without changing anything.'''
    def __init__(self, input_shape, itype, **kwargs):
        super(Print, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def forward(self, x):
        print('Forward\n -  x = %s'%x)
        return x
    
    @classmethod
    def backward(self, xy, dy, **kwargs):
        x,y=xy
        print('Backward\n -  x = %s\n -  y = %s\n -  dy = %s'%(x,y,dy))
        return EMPTY_VAR, dy

class Power(Function):
    '''
    Function x**order
    '''
    __display_attrs__ = ['order']
    def __init__(self, input_shape, itype, order, **kwargs):
        super(Power, self).__init__(input_shape, input_shape, itype)
        self.order = order

    def forward(self,x):
        return x**self.order

    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR, self.order*x**(self.order-1)*dy

class Cache(Function):
    '''Cache data without changing anything.'''
    def __init__(self, input_shape, itype, **kwargs):
        super(Cache, self).__init__(input_shape, input_shape, itype)
        self.forward_list = []
        self.backward_list = []

    def forward(self, x):
        self.forward_list.append(x)
        return x
    
    def backward(self, xy, dy, **kwargs):
        self.backward_list.append(dy)
        return EMPTY_VAR, dy

    def clear(self):
        self.forward_list = []
        self.backward_list = []

class Filter(Function):
    '''Momentum Filter.'''
    __display_attrs__ = ['momentum', 'axes']

    def __init__(self, input_shape, itype, momentum, axes, **kwargs):
        # sort axes and momentum
        DIM = len(input_shape)
        axes = [axis%DIM for axis in axes]
        order = np.argsort(axes)
        axes = tuple([axes[od] for od in order])
        momentum = np.atleast_1d(momentum)[order]

        self.momentum = momentum
        self.axes = axes
        size = [input_shape[axis] for axis in axes]
        self.filters = [np.exp(-1j*momentum*np.arange(ni)).reshape([1]*axis+[-1]+[1]*(DIM-axis-1))/ni for ki,axis,ni in zip(momentum,axes,size)]
        if all(momentum%np.pi==0):
            self.filters = [flt.real for flt in self.filters]
        output_shape = tuple([dim for axis,dim in enumerate(input_shape) if axis not in axes])
        #np.prod(np.ix_([np.exp(-1j*k*np.arange(ni))/ni for ki,ni in zip(momentum,size)]),axis=0)
        super(Filter, self).__init__(input_shape, output_shape, itype)

    def forward(self, x):
        y = x
        for axis, fltr in zip(self.axes[::-1], self.filters[::-1]):
            y = (y*fltr).sum(axis=axis)
        return y

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        dx = dy
        for axis, fltr in zip(self.axes, self.filters):
            dx = np.asarray(dx, order='F')[(slice(None),)*axis+(np.newaxis,)]*fltr.reshape(fltr.shape[:dx.ndim+1])
        return EMPTY_VAR, dx
