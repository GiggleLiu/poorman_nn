import numpy as np
from numbers import Number
import pdb

from core import Layer,Function, Tags, EXP_OVERFLOW, EMPTY_VAR
from lib.pooling import lib as fpooling
from lib.convprod import lib as fconvprod
from lib.relu import lib as frelu
from utils import scan2csc, tuple_prod

__all__=['Log2cosh','Sigmoid','Sum','Mul','Mod','Mean','ReLU','ConvProd','Pooling','DropOut','Sin','Cos','Power',
        'SoftMax','CrossEntropy','SoftMaxCrossEntropy','SquareLoss','Exp', 'Reshape','Transpose',
        'TypeCast', 'Print']

class Log2cosh(Function):
    '''
    Function log(2*cosh(theta)).
    '''
    def __init__(self, input_shape, dtype, **kwargs):
        super(Log2cosh, self).__init__(input_shape, input_shape, dtype)

    def forward(self,x):
        if np.ndim(x)==0:
            return np.log(2*np.cosh(x)) if abs(x.real)<=12 else np.sign(x.real)*x
        x=np.asarray(x)
        res=np.zeros_like(x)
        m1=x.real>EXP_OVERFLOW
        m2=x.real<-EXP_OVERFLOW
        m3=~(m1|m2)
        res[m1]=x[m1]
        res[m2]=-x[m2]
        res[m3]=np.log(2*np.cosh(x[m3]))
        return res.reshape(self.output_shape, order='F')

    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR(self.dtype),np.tanh(x)*dy

class Sigmoid(Function):
    '''
    Function log(2*cosh(theta)).
    '''
    def __init__(self, input_shape, dtype, **kwargs):
        super(Sigmoid, self).__init__(input_shape, input_shape, dtype)

    def forward(self,x):
        #for ndarray
        y=np.zeros_like(x)
        m1=x.real<-EXP_OVERFLOW
        m2=x.real>EXP_OVERFLOW
        m3=~(m1|m2)
        y[m2]=1
        y[m3]=1/(1+np.exp(-x[m3]))
        return y

    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR(self.dtype),y*(1-y)*dy

class Sum(Function):
    '''
    Sum along specific axis.
    '''
    def __init__(self, input_shape, dtype, axis, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        super(Sum,self).__init__(input_shape, output_shape, dtype)

    def forward(self,x):
        return np.sum(x,axis=self.axis)
    
    def backward(self, xy, dy, **kwargs):
        x, y = xy
        if np.ndim(dy)==0:
            dy_ = np.asarray(dy, order='F')[np.newaxis]
        else:
            dy_ = np.asarray(dy, order='F')[(slice(None),)*self.axis+(np.newaxis,)]
        dx = np.repeat(dy_,x.shape[self.axis],axis=self.axis)
        return EMPTY_VAR(self.dtype), dx

class Mean(Function):
    '''
    Mean along specific axis.
    '''
    def __init__(self,input_shape, dtype, axis, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        super(Mean,self).__init__(input_shape, output_shape, dtype)

    def forward(self,x):
        return np.mean(x,axis=self.axis)
    
    def backward(self, xy, dy, **kwargs):
        x, y = xy
        if np.ndim(dy)==0:
            dy_ = np.asarray(dy, order='F')[np.newaxis]
        else:
            dy_ = np.asarray(dy, order='F')[(slice(None),)*self.axis+(np.newaxis,)]
        dx = np.repeat(dy_,x.shape[self.axis],axis=self.axis)/x.shape[self.axis]
        return EMPTY_VAR(self.dtype), dx


class ReLU(Function):
    '''
    ReLU.
    '''
    def __init__(self, input_shape, dtype, leak = 0, is_inplace=False, **kwargs):
        super(ReLU,self).__init__(input_shape, input_shape, dtype, tags=Tags(runtimes=[], is_inplace=is_inplace))
        if leak>1 or leak<0:
            raise ValueError('leak parameter should be 0-1!')
        self.leak = leak

        #use the correct fortran subroutine.
        if dtype=='complex128':
            dtype_token = 'z'
        elif dtype=='complex64':
            dtype_token = 'c'
        elif dtype=='float64':
            dtype_token = 'd'
        elif dtype=='float32':
            dtype_token = 's'
        else:
            raise TypeError("data type error!")

        #use the correct function
        self._fforward=eval('frelu.forward_%s'%dtype_token)
        self._fbackward=eval('frelu.backward_%s'%dtype_token)

    def forward(self, x):
        y=self._fforward(x.ravel(order='F'),self.leak).reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        dx=self._fbackward(x=xy[0].ravel(order='F'),dy=dy.ravel(order='F'),leak=self.leak).reshape(self.input_shape, order='F')
        return EMPTY_VAR(self.dtype), dx

class Pooling(Function):
    '''
    Max/Mean pooling.

    Note:
        for complex numbers, what does max pooling looks like?
    '''
    mode_list = ['max', 'max-abs', 'min', 'min-abs', 'mean']

    def __init__(self, input_shape, dtype, kernel_shape, mode, boundary='O', **kwargs):
        self.kernel_shape = kernel_shape
        self.mode = mode
        if mode not in self.mode_list:
            raise ValueError('mode %s not allowed!'%mode)
        img_in_shape = input_shape[-len(kernel_shape):]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(kernel_shape, img_in_shape, strides=kernel_shape, boundary=boundary)
        output_shape = input_shape[:-len(kernel_shape)]+self.img_out_shape
        super(Pooling,self).__init__(input_shape, output_shape, dtype)

        #use the correct fortran subroutine.
        if dtype=='complex128':
            dtype_token = 'z'
        elif dtype=='complex64':
            dtype_token = 'c'
        elif dtype=='float64':
            dtype_token = 'd'
        elif dtype=='float32':
            dtype_token = 's'
        else:
            raise TypeError("data type error!")

        #use the correct function
        self._fforward=eval('fpooling.forward_%s'%dtype_token)
        self._fbackward=eval('fpooling.backward_%s'%dtype_token)

    def __repr__(self):
        return '<%s>(%s): %s -> %s'%(self.__class__.__name__,self.mode, self.input_shape,self.output_shape)

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
        return EMPTY_VAR(self.dtype), dx

class ConvProd(Function):
    '''
    Convolutional product layer.
    '''
    def __init__(self, input_shape, dtype, powers, strides=None, boundary='O', **kwargs):
        self.boundary = boundary
        self.powers = np.asarray(powers, order='F', dtype=dtype)

        img_nd = self.powers.ndim
        if strides is None:
            strides=(1,)*img_nd
        self.strides = strides

        img_in_shape = input_shape[-img_nd:]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(self.powers.shape, img_in_shape, strides=strides, boundary=boundary)
        output_shape = input_shape[:-img_nd]+self.img_out_shape
        super(ConvProd,self).__init__(input_shape, output_shape, dtype)

        #use the correct fortran subroutine.
        if dtype=='complex128':
            dtype_token = 'z'
        elif dtype=='complex64':
            dtype_token = 'c'
        elif dtype=='float64':
            dtype_token = 'd'
        elif dtype=='float32':
            dtype_token = 's'
        else:
            raise TypeError("data type error!")

        #use the correct function
        self._fforward=eval('fconvprod.forward_%s'%dtype_token)
        self._fbackward=eval('fconvprod.backward_%s'%dtype_token)

    def __repr__(self):
        return '<%s>: %s -> %s\n - strides = %s\n - filter = %s'%(self.__class__.__name__, self.input_shape,self.output_shape,self.strides,self.powers)

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
        return EMPTY_VAR(self.dtype), dx

class DropOut(Function):
    '''
    DropOut inplace.
    '''
    def __init__(self, input_shape, dtype, keep_rate, axis, is_inplace=False, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        self.keep_rate = keep_rate
        self.seed = None
        self.mask = None
        super(DropOut, self).__init__(input_shape, input_shape, dtype, tags=Tags(runtimes=['seed'],is_inplace=is_inplace))

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
        y=x if self.tags.is_inplace else x.copy(order='F')
        y[(slice(None),)*self.axis+(self.mask,)]/=self.keep_rate
        y[(slice(None),)*self.axis+(~self.mask,)]=0
        return y

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        dy[(slice(None),)*self.axis+(self.mask,)]/=self.keep_rate
        dy[(slice(None),)*self.axis+(~self.mask,)]=0
        return EMPTY_VAR(self.dtype), dy

class SoftMax(Function):
    '''
    Soft max function applied on the last axis.
    '''
    def __init__(self, input_shape, dtype, axis, **kwargs):
        self.axis=axis
        super(SoftMax, self).__init__(input_shape, input_shape, dtype)

    def forward(self, x):
        x=x-x.max(axis=self.axis, keepdims=True)
        rho=np.exp(x)
        return rho/rho.sum(axis=self.axis, keepdims=True)

    def backward(self, xy, dy, **kwargs):
        x,y = xy
        return EMPTY_VAR(self.dtype),dy*y-(dy*y).sum(axis=self.axis, keepdims=True)*y

class CrossEntropy(Function):
    '''
    Cross Entropy sum(p*log(q)). With p the true labels.
        q = x
    '''
    ZERO_REF=1e-15

    def __init__(self, input_shape, dtype, axis, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        self.y_true = None
        output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        super(CrossEntropy, self).__init__(input_shape, output_shape, dtype, tags=Tags(runtimes=['y_true'], is_inplace=False))

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, note 0 < x <= 1.
            :y_true: ndarray, correct one-hot y.
        '''
        return (-self.y_true*np.log(np.maximum(self.ZERO_REF,x))).sum(axis=self.axis)

    def backward(self, xy, dy, **kwargs):
        x,y = xy
        return EMPTY_VAR(self.dtype),-dy[(slice(None),)*self.axis+(np.newaxis,)]*(self.y_true/np.maximum(x, self.ZERO_REF))

class SoftMaxCrossEntropy(Function):
    '''
    Cross Entropy sum(p*log(q)). With p the true labels.
        q = exp(x)/sum(exp(x))
    '''
    def __init__(self, input_shape, dtype, axis, **kwargs):
        if axis > len(input_shape)-1: raise ValueError('invalid axis')
        self.axis=axis%len(input_shape)
        self.y_true = None
        output_shape = input_shape[:axis]+input_shape[self.axis+1:]
        super(SoftMaxCrossEntropy, self).__init__(input_shape, output_shape, dtype, tags=Tags(runtimes=['y_true'], is_inplace=False))

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
        return ((np.log(Z)-x)*self.y_true).sum(axis=self.axis)

    def backward(self, xy, dy, **kwargs):
        x,y = xy
        x=x-x.max(axis=self.axis, keepdims=True)
        rho=np.exp(x)
        Z=rho.sum(axis=self.axis, keepdims=True)
        y1=rho/Z
        return EMPTY_VAR(self.dtype),dy[(slice(None),)*self.axis+(np.newaxis,)]*(y1-self.y_true)

class SquareLoss(Function):
    '''
    Square Loss (p-q)**2. With p the true labels.
    '''
    def __init__(self, input_shape, dtype, **kwargs):
        self.y_true = None
        super(SquareLoss, self).__init__(input_shape, input_shape, dtype, tags=Tags(runtimes=['y_true'], is_inplace=False))

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
        is_complex = self.dtype[:7]=='complex'
        #return EMPTY_VAR(self.dtype),((xy[0]-xt).conj()*dy) if self.dtype[:7]=='complex' else (2*(xy[0]-xt)*dy)
        return EMPTY_VAR(self.dtype),((xy[0]-xt)*dy) if self.dtype[:7]=='complex' else (2*(xy[0]-xt)*dy)  #paritial x.conj() for complex

class Exp(Function):
    '''
    Function exp(x)
    '''
    def __init__(self, input_shape, dtype, **kwargs):
        super(Exp, self).__init__(input_shape, input_shape, dtype)

    def forward(self,x):
        return np.exp(x)

    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR(self.dtype),dy*y

class Sin(Function):
    '''
    Function sin(x)
    '''
    def __init__(self, input_shape, dtype, **kwargs):
        super(Sin, self).__init__(input_shape, input_shape, dtype)

    def forward(self,x):
        return np.sin(x)

    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR(self.dtype),dy*np.cos(x)

class Cos(Function):
    '''
    Function cos(x)
    '''
    def __init__(self, input_shape, dtype, **kwargs):
        super(Cos, self).__init__(input_shape, input_shape, dtype)

    def forward(self,x):
        return np.cos(x)

    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR(self.dtype),-dy*np.sin(x)

class Reshape(Function):
    def forward(self, x):
        return x.reshape(self.output_shape, order='F')

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR(self.dtype), dy.reshape(self.input_shape, order='F')

class TypeCast(Function):
    def __init__(self, input_shape, dtype, otype, **kwargs):
        super(TypeCast, self).__init__(input_shape, input_shape, dtype, otype=otype)

    def forward(self, x):
        return np.asarray(x, dtype=self.otype, order='F')

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR(self.dtype), np.asarray(dy, dtype=self.dtype, order='F')

class Transpose(Function):
    def __init__(self, input_shape, dtype, axes, **kwargs):
        self.axes=axes
        if len(axes)!=len(input_shape):
            raise ValueError('axes incorrect!')
        output_shape=tuple([input_shape[axis] for axis in self.axes])
        super(Transpose, self).__init__(input_shape, output_shape, dtype)

    def forward(self, x):
        return x.transpose(self.axes)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        return EMPTY_VAR(self.dtype), dy.transpose(np.argsort(self.axes))

class Mul(Function):
    '''Multiply by a constant'''
    def __init__(self, input_shape, dtype, alpha, **kwargs):
        self.alpha = alpha
        super(Mul, self).__init__(input_shape, input_shape, dtype)

    def forward(self, x): return self.alpha*x
    def backward(self,xy, dy, **kwargs): return EMPTY_VAR(self.dtype), self.alpha*dy

class Mod(Function):
    '''Mod by a constant'''
    def __init__(self, input_shape, dtype, n, **kwargs):
        self.n = n
        super(Mod, self).__init__(input_shape, input_shape, dtype)

    def forward(self, x): return x%self.n
    def backward(self,xy, dy, **kwargs): return EMPTY_VAR(self.dtype), dy

class Print(Function):
    '''Print data without changing anything.'''
    def __init__(self, input_shape, dtype, **kwargs):
        super(Print, self).__init__(input_shape, input_shape, dtype)

    def forward(self, x):
        print 'Forward\n -  x = %s'%x
        return x
    
    def backward(self, xy, dy, **kwargs):
        x,y=xy
        print 'Backward\n -  x = %s\n -  y = %s\n -  dy = %s'%(x,y,dy)
        return EMPTY_VAR(self.dtype), dy

class Power(Function):
    '''
    Function x**order
    '''
    def __init__(self, input_shape, dtype, order, **kwargs):
        super(Power, self).__init__(input_shape, input_shape, dtype)
        self.order = order

    def forward(self,x):
        return x**self.order

    def backward(self,xy,dy, **kwargs):
        x, y = xy
        return EMPTY_VAR(self.dtype), self.order*x**(self.order-1)*dy

