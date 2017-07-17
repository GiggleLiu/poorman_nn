import numpy as np
from numpy import ma
import pdb

from core import Layer,Function,RandFunction,SupervisedLayer
from utils import scan2csc

__all__=['Log2cosh','Sigmoid','Sum','ReLU','MaxPool','DropOut',
        'SoftMax','CrossEntropy','SoftMaxCrossEntropy']

EXP_OVERFLOW=30

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
        return (),np.tanh(x)*dy

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
        return (),y*(1-y)*dy

class Reorder(Function):
    '''
    Switch order of variables.
    '''
    def __init__(self,order):
        self.order=order

    def forward(self,x):
        return np.transpose(x,axes=self.order)

    def backward(self,x,y,dy):
        return (),np.transpose(dy,axes=argsort(self.order))

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
        return (),[dy[self.cumdims[i]:self.cumdims[i+1]] for i in xrange(len(self.cumdims)-1)]

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
        return (),np.repeat(dy_,self._nitem,axis=self.axis)


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
        return (), dx

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
            :x: ndarray, (num_batch, nfi, img_in_dims), input in 'F' order.

        Return:
            ndarray, (num_batch, nfo, img_out_dims), output in 'F' order.
        '''
        x_nd, img_nd = x.ndim, self.img_nd
        x=x.reshape(x.shape[:x_nd-img_nd]+(np.prod(self.img_in_shape),), order='F')
        y=np.empty(x.shape[:x_nd-img_nd]+(np.prod(self.img_out_shape),), order='F')
        for col,(start,end) in enumerate(zip(self.csc_indptr[:-1],self.csc_indptr[1:])):
            indices=self.csc_indices[start-1:end-1]-1
            y[...,col]=np.max(x[...,indices],keepdims=True,axis=-1)
        y=y.reshape(y.shape[:x_nd-img_nd]+self.img_out_shape,order='F')
        return y

    def backward(self, x, y, dy):
        '''It will shed a mask on dy'''
        x_nd, img_nd = x.ndim, self.img_nd
        pre_nd=x_nd-img_nd
        dim_in=np.prod(self.img_in_shape)
        dim_out=np.prod(self.img_out_shape)
        x_=x.reshape(x.shape[:pre_nd]+(dim_in,),order='F')
        y=y.reshape(y.shape[:pre_nd]+(dim_out,),order='F')
        dy=dy.reshape(dy.shape[:pre_nd]+(dim_out,),order='F')
        #dx=ma.zeros(x_.shape,fill_value=0,dtype=dy.dtype)
        #dx.mask=np.ones(dx.shape,dtype='bool')
        dx=np.zeros(x_.shape,dtype=dy.dtype)
        preinds=np.indices(dx.shape[:pre_nd])
        for col,(start,end) in enumerate(zip(self.csc_indptr[:-1],self.csc_indptr[1:])):
            indices=self.csc_indices[start-1:end-1]-1
            maxind=indices[np.argmax(x_[...,indices],axis=-1)]
            #dx.mask[maxind,ind1,ind2]=False
            #dx.data[maxind,ind1,ind2]=dyi
            dx[tuple(preinds)+(maxind,)]=dy[...,col]
        return (), dx.reshape(x.shape, order='F')

class DropOut(RandFunction):
    '''
    DropOut.
    '''
    def __init__(self, keep_rate):
        super(RandFunction, self).__init__()
        self.keep_rate = keep_rate

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (num_feature_in, num_batch), in fortran order.
        '''
        #generate a random state, a keep mask
        self.state = np.random.random(x.shape[0])<self.keep_rate
        return x[self.state,:]

    def backward(self, x, y, dy):
        dx=np.zeros_like(x)
        dx[self.state] = dy
        return (), dx

class SoftMax(Function):
    '''
    Soft max function applied on the last axis.
    '''
    def forward(self, x):
        x=x-x.max(axis=-1, keepdims=True)
        rho=np.exp(x)
        return rho/rho.sum(axis=-1, keepdims=True)

    def backward(self, x, y, dy):
        return (),dy*y*(1-y)

class CrossEntropy(Function, SupervisedLayer):
    '''
    Cross Entropy sum(p*log(q)). With p the true labels.
        q = x
    '''
    def forward(self, x, y_true):
        '''
        Parameters:
            :x: ndarray, note 0 < x <= 1.
            :y_true: ndarray, correct one-hot y.
        '''
        return (-y_true*np.log(x)).sum(axis=-1)

    def backward(self, x, y, dy, y_true):
        return (),-dy[...,np.newaxis]*(y_true/x)

class SoftMaxCrossEntropy(Function, SupervisedLayer):
    '''
    Cross Entropy sum(p*log(q)). With p the true labels.
        q = exp(x)/sum(exp(x))
    '''
    def forward(self, x, y_true):
        '''
        Parameters:
            :x: ndarray, note 0 < x <= 1.
            :y_true: ndarray, correct one-hot y.
        '''
        x=x-x.max(axis=-1, keepdims=True)
        rho=np.exp(x)
        Z=rho.sum(axis=-1, keepdims=True)
        return ((np.log(Z)-x)*y_true).sum(axis=-1)

    def backward(self, x, y, dy, y_true):
        x=x-x.max(axis=-1, keepdims=True)
        rho=np.exp(x)
        Z=rho.sum(axis=-1, keepdims=True)
        y1=rho/Z
        return (),dy[...,np.newaxis]*y_true*(y1-1)

