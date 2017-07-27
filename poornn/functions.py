import numpy as np
from numbers import Number
import pdb

from core import Layer,Function, SupervisedLayer, Tags, EXP_OVERFLOW, EMPTY_VAR, check_shape
from utils import scan2csc

__all__=['Log2cosh','Sigmoid','Sigmoid_I','Sum','Mean','ReLU_I','Pooling','DropOut_I',
        'SoftMax','CrossEntropy','SoftMaxCrossEntropy','Exp', 'Reshape','Transpose']

class Log2cosh(Function):
    '''
    Function log(2*cosh(theta)).
    '''
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

    def backward(self,x,y,dy, **kwargs):
        return EMPTY_VAR(x.dtype),np.tanh(x)*dy

class Sigmoid_I(Function):
    '''
    Function log(2*cosh(theta)).
    '''
    tags = Tags(is_runtime = False, is_inplace = True)
    def forward(self,x):
        #for ndarray
        m1=x.real<-EXP_OVERFLOW
        m2=x.real>EXP_OVERFLOW
        m3=~(m1|m2)
        x[m1]=0
        x[m2]=1
        x[m3]=1/(1+np.exp(-x[m3]))
        return x

    def backward(self,x,y,dy, **kwargs):
        raise Exception('Inplace backward should change x!')
        return EMPTY_VAR(x.dtype),y*(1-y)*dy

class Sigmoid(Function):
    '''
    Function log(2*cosh(theta)).
    '''
    def forward(self,x):
        #for ndarray
        y=np.zeros_like(x)
        m1=x.real<-EXP_OVERFLOW
        m2=x.real>EXP_OVERFLOW
        m3=~(m1|m2)
        y[m2]=1
        y[m3]=1/(1+np.exp(-x[m3]))
        return y

    def backward(self,x,y,dy, **kwargs):
        return EMPTY_VAR(x.dtype),y*(1-y)*dy


class Reorder(Function):  #TODO: fix initialization
    '''
    Switch order of variables.
    '''
    def __init__(self,order, *args, **kwargs):
        self.order=order
        super(Reorder, self).__init__(*args, **kwargs)

    def forward(self,x):
        return np.transpose(x,axes=self.order)

    def backward(self,x,y,dy, **kwargs):
        return EMPTY_VAR(x.dtype),np.transpose(dy,axes=argsort(self.order))

class Merge(Function):
    '''
    Merge x and x' into a single array.
    
    Needed?
    '''
    def __init__(self,order, *args, **kwargs):
        self.cumdims=None
        super(Reorder, self).__init__(*args, **kwargs)

    def forward(self,x):
        '''
        x is a list of inputs.
        '''
        self.cumdims=np.append([0],np.cumsum([xi.shape for xi in x]))
        return np.concatenate(x)
    
    def backward(self,x,y,dy, **kwargs):
        return EMPTY_VAR(x.dtype),[dy[self.cumdims[i]:self.cumdims[i+1]] for i in xrange(len(self.cumdims)-1)]

class Sum(Function):
    '''
    Sum along specific axis.
    '''
    def __init__(self, input_shape, axis, output_shape=None, dtype='float32'):
        self.axis=axis%len(input_shape)
        if output_shape is None:
            output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        super(Sum,self).__init__(input_shape, output_shape, dtype='float32')

    def forward(self,x):
        return np.sum(x,axis=self.axis)
    
    def backward(self,x,y,dy, **kwargs):
        dy_=np.asfortranarray(dy)[(slice(None),)*self.axis+(np.newaxis,)]
        return EMPTY_VAR(x.dtype),np.repeat(dy_,x.shape[self.axis],axis=self.axis)

class Mean(Function):
    '''
    Mean along specific axis.
    '''
    def __init__(self,input_shape, axis, output_shape=None, dtype='float32'):
        self.axis=axis%len(input_shape)
        if output_shape is None:
            output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        super(Mean,self).__init__(input_shape, output_shape, dtype='float32')

    def forward(self,x):
        return np.mean(x,axis=self.axis)
    
    def backward(self,x,y,dy, **kwargs):
        dy_=dy[(slice(None),)*self.axis+(np.newaxis,)]
        return EMPTY_VAR(x.dtype),np.repeat(dy_,x.shape[self.axis],axis=self.axis)/x.shape[self.axis]

class ReLU_I(Function):
    '''
    ReLU.
    '''
    tags = Tags(is_runtime = False, is_inplace = True)
    def __init__(self, leak = 0, input_shape=None, output_shape=None, dtype='float32'):
        super(ReLU_I,self).__init__(input_shape, output_shape, dtype='float32')
        if leak>1 or leak<0:
            raise ValueError('leak parameter should be 0-1!')
        self.leak = leak

    def forward(self, x):
        xmask=x<=0
        if self.leak==0:
            x[xmask]=0
        else:
            x[xmask]*=self.leak
        return x

    def backward(self, x, y, dy, **kwargs):
        xmask=y<=0
        if self.leak==0:
            dy[xmask]=0
        else:
            dy[xmask]*=self.leak
        return EMPTY_VAR(x.dtype), dy

class Pooling(Function):
    '''
    Max/Mean pooling.

    Note:
        for complex numbers, what does max pooling looks like?
    '''
    mode_dict = {'max':np.max,
            'mean':np.mean}

    def __init__(self, input_shape, kernel_shape, mode, boundary='O', output_shape=None, dtype='float32'):
        self.kernel_shape = kernel_shape
        self.mode = mode
        if not self.mode_dict.has_key(mode):
            raise ValueError('mode %s not allowed!'%mode)
        img_in_shape = input_shape[-len(kernel_shape):]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(kernel_shape, img_in_shape, strides=kernel_shape, boundary=boundary)
        if output_shape is None:
            output_shape = input_shape[:-len(kernel_shape)]+self.img_out_shape
        super(Pooling,self).__init__(input_shape, output_shape, dtype='float32')

    def __repr__(self):
        return '<%s>(%s): %s -> %s'%(self.__class__.__name__,self.mode, self.input_shape,self.output_shape)

    @property
    def img_nd(self):
        return len(self.kernel_shape)

    @property
    def pooling_func(self):
        '''Pooling Function.'''
        return self.mode_dict[self.mode]

    @check_shape((1,))
    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), input in 'F' order.

        Return:
            ndarray, (num_batch, nfo, img_out_dims), output in 'F' order.
        '''
        x_nd, img_nd = x.ndim, self.img_nd
        pooling_func = self.pooling_func
        x=x.reshape(x.shape[:x_nd-img_nd]+(-1,), order='F')
        y=np.empty(x.shape[:x_nd-img_nd]+(np.product(self.img_out_shape),), order='F', dtype=x.dtype)
        for col,(start,end) in enumerate(zip(self.csc_indptr[:-1],self.csc_indptr[1:])):
            indices=self.csc_indices[start-1:end-1]-1
            y[...,col]=pooling_func(x[...,indices],keepdims=False,axis=-1)
        y=y.reshape(self.output_shape, order='F')
        return y

    @check_shape((1, -3))
    def backward(self, x, y, dy, **kwargs):
        '''It will shed a mask on dy'''
        x_nd, img_nd = x.ndim, self.img_nd
        xpre=x.shape[:x_nd-img_nd]
        yshape=xpre[:-1]+(-1,np.prod(self.img_out_shape))

        #flatten inputs/outputs
        x_=x.reshape(xpre+(-1,), order='F')
        dy=dy.reshape(yshape, order='F')

        dx=np.zeros_like(x_)
        preinds=np.indices(dx.shape[:x_nd-img_nd])
        if self.mode == 'max':
            for col,(start,end) in enumerate(zip(self.csc_indptr[:-1],self.csc_indptr[1:])):
                indices=self.csc_indices[start-1:end-1]-1
                maxind=indices[np.argmax(x_[...,indices],axis=-1)]
                dx[tuple(preinds)+(maxind,)]=dy[...,col]
        else:
            kernel_size = np.prod(self.kernel_shape)
            for col,(start,end) in enumerate(zip(self.csc_indptr[:-1],self.csc_indptr[1:])):
                indices=self.csc_indices[start-1:end-1]-1
                dy_=dy[...,col]/kernel_size
                for index in indices:
                    dx[tuple(preinds)+(index,)]=dy_
        return EMPTY_VAR(x.dtype), dx.reshape(x.shape, order='F')

class DropOut_I(Function):
    '''
    DropOut inplace.
    '''
    tags = Tags(is_runtime = True, is_inplace = True)
    def __init__(self, keep_rate, axis, input_shape=None, output_shape=None, dtype='float32'):
        self.axis=axis%len(input_shape)
        self.keep_rate = keep_rate
        super(DropOut_I, self).__init__(input_shape, output_shape, dtype='float32')

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (num_batch, num_feature_in), in fortran order.
        '''
        #generate a random state, a keep mask
        self.state = np.random.random(x.shape[self.axis])<self.keep_rate
        x[(slice(None),)*self.axis+(self.state,)]/=self.keep_rate
        x[(slice(None),)*self.axis+(~self.state,)]=0
        return x

    def backward(self, x, y, dy, **kwargs):
        dy[(slice(None),)*self.axis+(self.state,)]/=self.keep_rate
        dy[(slice(None),)*self.axis+(~self.state,)]=0
        return EMPTY_VAR(x.dtype), dy

class SoftMax(Function):
    '''
    Soft max function applied on the last axis.
    '''
    def __init__(self, input_shape, axis, output_shape = None, dtype='float32'):
        self.axis=axis
        if output_shape is None:
            output_shape = input_shape
        super(SoftMax, self).__init__(input_shape, output_shape, dtype='float32')

    def forward(self, x):
        x=x-x.max(axis=self.axis, keepdims=True)
        rho=np.exp(x)
        return rho/rho.sum(axis=self.axis, keepdims=True)

    def backward(self, x, y, dy, **kwargs):
        return EMPTY_VAR(x.dtype),dy*y-(dy*y).sum(axis=self.axis, keepdims=True)*y
        #return EMPTY_VAR(x.dtype),dy*y*(1-y.sum(axis=self.axis, keepdims=True))

class CrossEntropy(Function, SupervisedLayer):
    '''
    Cross Entropy sum(p*log(q)). With p the true labels.
        q = x
    '''
    ZERO_REF=1e-15

    def __init__(self, input_shape, axis, output_shape = None, dtype='float32'):
        self.axis=axis%len(input_shape)
        if output_shape is None:
            output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        super(CrossEntropy, self).__init__(input_shape, output_shape, dtype='float32')

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, note 0 < x <= 1.
            :y_true: ndarray, correct one-hot y.
        '''
        return (-self.y_true*np.log(np.maximum(self.ZERO_REF,x))).sum(axis=self.axis)

    def backward(self, x, y, dy, **kwargs):
        return EMPTY_VAR(x.dtype),-dy[(slice(None),)*self.axis+(np.newaxis,)]*(self.y_true/np.maximum(x, self.ZERO_REF))

class SoftMaxCrossEntropy(Function, SupervisedLayer):
    '''
    Cross Entropy sum(p*log(q)). With p the true labels.
        q = exp(x)/sum(exp(x))
    '''
    def __init__(self, input_shape, axis, output_shape = None, dtype='float32'):
        self.axis=axis%len(input_shape)
        if output_shape is None:
            output_shape = input_shape[:axis]+input_shape[self.axis+1:]
        super(SoftMaxCrossEntropy, self).__init__(input_shape, output_shape, dtype='float32')

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, note 0 < x <= 1.
            :y_true: ndarray, correct one-hot y.
        '''
        x=x-x.max(axis=self.axis, keepdims=True)
        rho=np.exp(x)
        Z=rho.sum(axis=self.axis, keepdims=True)
        return ((np.log(Z)-x)*self.y_true).sum(axis=self.axis)

    def backward(self, x, y, dy, **kwargs):
        x=x-x.max(axis=self.axis, keepdims=True)
        rho=np.exp(x)
        Z=rho.sum(axis=self.axis, keepdims=True)
        y1=rho/Z
        return EMPTY_VAR(x.dtype),dy[(slice(None),)*self.axis+(np.newaxis,)]*(y1-self.y_true)

class Exp(Function):
    '''
    Function exp(x)
    '''
    def forward(self,x):
        return np.exp(x)

    def backward(self,x,y,dy, **kwargs):
        return EMPTY_VAR(x.dtype),dy*y

class Reshape(Function):
    def forward(self, x):
        return x.reshape(self.output_shape)

    def backward(self, x, y, dy, **kwargs):
        return EMPTY_VAR(x.dtype), dy.reshape(self.input_shape)


class Transpose(Function):
    def __init__(self, input_shape, axes, output_shape=None, dtype='float32'):
        self.axes=axes
        if len(axes)!=len(input_shape):
            raise ValueError('axes incorrect!')
        if output_shape is None:
            output_shape=tuple([input_shape[axis] for axis in self.axes])
        super(Transpose, self).__init__(input_shape, output_shape, dtype='float32')

    def forward(self, x):
        return x.transpose(self.axes)

    def backward(self, x, y, dy, **kwargs):
        return EMPTY_VAR(x.dtype), dy.transpose(np.argsort(self.axes))
