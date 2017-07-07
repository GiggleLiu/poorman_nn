import numpy as np

from core import Layer,EMPTY_ARRAY

__all__=['Function','Log2cosh','Sigmoid']

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

#TODO: sum, maxpool?, onehot?
