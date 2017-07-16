import numpy as np
from numpy import ma
import pdb

from core import Layer,EMPTY_ARRAY
from utils import scan2csc

__all__=['Function','Log2cosh','Sigmoid','Sum','ReLU','PReLU','MaxPool']

EXP_OVERFLOW=30

class Function(Layer):
    '''Function layer with no variables.'''
    def __call__(self,x):
        return self.forward(x)

    def get_variables(self):
        return EMPTY_ARRAY

    def set_variables(self,*args,**kwargs):
        pass

class Log2cosh(Function):
    '''
    Function log(2*cosh(theta)).
    '''
    def forward(self,x):
        if np.ndim(x)==0:
            return np.log(2*np.cosh(x)) if abs(x.real)<=12 else np.sign(x.real)*x
        x=np.asarray(x)
        res=np.zeros_like(x)
        overflow=abs(x.real)>EXP_OVERFLOW
        to=x[overflow]
        res[overflow]=np.sign(to.real)*to
        res[~overflow]=np.log(2*np.cosh(x[~overflow]))
        return res

    def backward(self,x,y,dy=1.):
        return EMPTY_ARRAY,np.tanh(x)*dy

class Sigmoid(Function):
    '''
    Function log(2*cosh(theta)).
    '''
    def forward(self,x):
        #for number
        if np.ndim(x)==0:
            if x.real<=-EXP_OVERFLOW:
                return 0.
            elif x.real>=EXP_OVERFLOW:
                return 1.
            else:
                return 1/(1+np.exp(-x))

        #for ndarray
        x=np.asarray(x)
        res=np.zeros_like(x)
        res[x.real<-EXP_OVERFLOW]=0
        res[x.real>EXP_OVERFLOW]=1
        mm=abs(x.real)<=EXP_OVERFLOW
        res[mm]=1/(1+np.exp(-x[mm]))
        return res

    def backward(self,x,y,dy=1.):
        y=np.asarray(y)
        return EMPTY_ARRAY,y*(1-y)*dy

class Reorder(Function):
    '''
    Switch order of variables.
    '''
    def __init__(self,order):
        self.order=order

    def forward(self,x):
        return np.transpose(x,axes=self.order)

    def backward(self,x,y,dy):
        return EMPTY_ARRAY,np.transpose(dy,axes=argsort(self.order))

class Merge(Function):
    '''
    Merge x and x' into a single array.
    
    Needed?
    '''
    def __init__(self):
        self.cumdims=None

    def forward(self,x):
        '''
        x is a list of inputs.
        '''
        self.cumdims=np.append([0],np.cumsum([xi.shape for xi in x]))
        return np.concatenate(x)
    
    def backward(self,x,y,dy):
        return EMPTY_ARRAY,[dy[self.cumdims[i]:self.cumdims[i+1]] for i in xrange(len(self.cumdims)-1)]

class Sum(Function):
    '''
    Sum along specific axis.
    '''
    def __init__(self,axis):
        self.axis=axis
        self._nitem=None

    def forward(self,x):
        self._nitem=x.shape[self.axis]
        return np.sum(x,axis=self.axis)
    
    def backward(self,x,y,dy):
        dy_=dy[(slice(None),)*self.axis+(np.newaxis,)]
        return EMPTY_ARRAY,np.repeat(dy_,self._nitem,axis=self.axis)


class ReLU(Function):
    '''
    ReLU.
    '''
    def __init__(self, leak = 0):
        self.leak = leak
        if leak>1 or leak<0:
            raise ValueError('leak parameter should be 0-1!')

    def forward(self, x):
        if self.leak==0:
            return maximum(x,0)
        else:
            return maximum(x,self.leak*x)

    def backward(self, x, y, dy):
        dx=dy.copy()
        if self.leak==0:
            dx[x<0]=0
        else:
            dx[x<0]=leak*dy
        return EMPTY_ARRAY, dx

class PReLU(Function):
    '''
    Parametric ReLU.
    '''
    def __init__(self, leak = 0, dtype='float64'):
        self.leak = leak
        self.dtype=dtype
        if leak>1 or leak<0:
            raise ValueError('leak parameter should be 0-1!')

    def forward(self, x):
        if self.leak==0:
            return maximum(x,0)
        else:
            return maximum(x,self.leak*x)

    def backward(self, x, y, dy):
        dx=dy.copy()
        mask=x<0
        if self.leak==0:
            dx[mask]=0
        else:
            dx[mask]=leak*dy
        da = np.sum(dy[mask]*x[mask].conj())
        return np.array([da], dtype=self.dtype), dx

    def get_variables(self):
        return np.array([self.leak], dtype=self.dtype)

    def set_variables(self, a):
        self.leak=a.item()

class MaxPool(Function):
    '''
    Max pooling.

    Note:
        for complex numbers, what does max pooling looks like?
    '''
    def __init__(self, kernel_shape, img_in_shape, boundary):
        self.kernel_shape = kernel_shape
        self.img_in_shape = img_in_shape
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(kernel_shape, img_in_shape, strides=kernel_shape, boundary=boundary)

    @property
    def img_nd(self):
        return len(self.kernel_shape)

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (dim_in(s), num_feature_in, num_batch), input in 'C' order.

        Return:
            ndarray, (dim_out(s), num_feature_out, num_batch), output in 'C' order.
        '''
        res=[]
        x=x.reshape((-1,)+x.shape[-2:])
        for start,end in zip(self.csc_indptr[:-1],self.csc_indptr[1:]):
            indices=self.csc_indices[start-1:end-1]-1
            res.append(np.max(x[indices],axis=0))
        res=np.reshape(res,self.img_out_shape+x.shape[-2:])
        return res

    def backward(self, x, y, dy):
        '''It will shed a mask on dy'''
        x_=x.reshape((-1,)+x.shape[-2:])
        dy=dy.reshape((-1,)+dy.shape[-2:])
        dx=ma.zeros(x_.shape,fill_value=0,dtype=dy.dtype)
        dx.mask=np.ones(dx.shape,dtype='bool')
        ind1,ind2=np.indices(x.shape[-2:])
        for dyi,start,end in zip(dy,self.csc_indptr[:-1],self.csc_indptr[1:]):
            indices=self.csc_indices[start-1:end-1]-1
            maxind=indices[np.argmax(x_[indices],axis=0)]
            dx.mask[maxind,ind1,ind2]=False
            dx.data[maxind,ind1,ind2]=dyi
        return EMPTY_ARRAY, dx.reshape(x.shape)

#TODO: onehot?
