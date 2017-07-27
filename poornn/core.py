'''
ABC of neural network.
'''

import numpy as np
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import pdb

from utils import typed_random

__all__=['Layer','Function','SupervisedLayer','ANN','check_shape', 'check_numdiff',
        'Tags', 'EXP_OVERFLOW', 'EMPTY_VAR']

'''
Attributes for Tags:
    :is_runtime: bool, True if the layer uses RNG, or some runtime variables that make forward and backward dependant.
    :is_inplace: bool, True if the output is made by changing input inplace.
'''

Tags = namedtuple('Tags',('is_runtime', 'is_inplace'))
EXP_OVERFLOW = 12
EMPTY_VAR = lambda dtype: np.zeros([0], dtype=dtype)

class Layer(object):
    '''
    A single layer in Neural Network.
    '''

    __metaclass__ = ABCMeta
    tags = Tags(is_runtime = False, is_inplace = False)

    def __init__(self, input_shape, output_shape, dtype='float32'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dtype = dtype

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s>: %s -> %s'%(self.__class__.__name__,self.input_shape,self.output_shape)

    def _check_input(self, x):
        if self.input_shape is None:
            return
        if x.ndim!=len(self.input_shape):
            raise ValueError('Dimension mismatch! x %s, desire %s'%(x.ndim, len(self.input_shape)))
        for shape_i, xshape_i in zip(self.input_shape, x.shape):
            if shape_i!=-1 and shape_i!=xshape_i:
                raise ValueError('Illegal Input shape! x %s, desire %s'%(x.shape, self.input_shape))

    def _check_output(self, y):
        if self.output_shape is None:
            return
        if y.ndim!=len(self.output_shape):
            raise ValueError('Dimension mismatch! y %s, desire %s'%(y.ndim, len(self.output_shape)))
        for shape_i, yshape_i in zip(self.output_shape, y.shape):
            if shape_i!=-1 and shape_i!=yshape_i:
                raise ValueError('Illegal Output shape! y %s, desire %s'%(y.shape, self.output_shape))

    @abstractmethod
    def forward(self,x):
        '''
        Forward propagration to evaluate F(x).

        Parameters:
            :x: ndarray, input array.

        Return:
            ndarray, output array y.
        '''
        pass

    @abstractmethod
    def backward(self,x,y,dy,mask=(1,1)):
        '''
        Back propagation.

        Parameters:
            :x: ndarray, input array.
            :y: ndarray, output array.
            :dy: ndarray, derivative of cost with respect to output array.
            :mask: tuple, (do_wgrad, do_xgrad)

        Return:
            (ndarray, ndarray), \partial J/\partial V_f and \partial J/\partial x.
        '''
        pass

    @abstractmethod
    def get_variables(self):
        '''
        Get current variables.

        Return:
            1darray,
        '''
        pass

    @abstractmethod
    def set_variables(self, variables, mode='set'):
        '''
        Change current variables.

        Parameters:
            :variables: 1darray,
            :mode: choice('set', 'add').
        '''
        pass

    @property
    @abstractmethod
    def num_variables(self):
        '''Number of variables.'''
        pass

class Function(Layer):
    '''Function layer with no variables.'''
    def __init__(self, input_shape = None, output_shape = None, dtype = 'float32'):
        if output_shape is None:
            output_shape = input_shape
        super(Function, self).__init__(input_shape, output_shape, dtype = dtype)

    def __call__(self,x):
        return self.forward(x)

    def get_variables(self):
        return EMPTY_VAR(self.dtype)

    def set_variables(self,*args,**kwargs):
        pass

    @property
    def num_variables(self):
        return 0

class SupervisedLayer(Layer):
    '''A special layer used in supervised learning. Often used as cost function.'''
    def __init__(self, *args, **kwargs):
        super(SupervisedLayer,self).__init__(*args, **kwargs)
        self.y_true = None

    def set_y_true(self, y_true):
        '''Set labels.'''
        self.y_true = y_true

class ANN(object):
    '''
    Artificial Neural network state.
    '''
    def __init__(self,layers):
        #TODO: check multiple use of runtime layers.
        self.layers=layers

        #check connections
        for la,lb in zip(layers[:-1], layers[1:]):
            if lb.input_shape==None:
                lb.input_shape=la.output_shape
                if lb.output_shape is None:
                    lb.output_shape=la.output_shape
                continue
            #check input and output shape
            for sa, sb in zip(la.output_shape, lb.input_shape):
                if sa!=sb:
                    if sa==-1:
                        sa=sb
                    elif sb==-1:
                        sb=sa
                    else:
                        raise Exception('Shape between layers(%s,%s) mismatch! in%s, out%s'%(la,lb, la.output_shape, lb.input_shape))

    @property
    def num_layers(self):
        return len(self.layers)

    def feed_input(self, x, y_true=None):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), in 'F' order.

        Return:
            list, output in each layer.
        '''
        ys=[x]
        for layer in self.layers:
            if isinstance(layer, SupervisedLayer):
                layer.set_y_true(y_true)
            x=layer.forward(x)
            ys.append(x)
        return ys

    def back_propagate(self,ys, dy=np.array(1)):
        '''
        Compute gradients.

        Parameters:
            :ys: list, output in each layer.

        Return:
            list, gradients for vairables in layers.
        '''
        dvs=[]
        x_broken = False
        for i in xrange(1,len(ys)):
            x, y = ys[-i-1], ys[-i]
            layer = self.layers[-i]
            dv, dy=layer.backward(x, None if x_broken else y, dy, mask=(1,1))
            dvs.append(dv)
            x_broaken = layer.tags.is_inplace
        return np.concatenate(dvs[::-1]), dy

    def get_variables(self):
        '''Dump values to an array.'''
        return np.concatenate([layer.get_variables() for layer in self.layers])

    def set_variables(self,v, mode='set'):
        '''
        Load data from an array.
        
        Parameters:
            :v: 1darray, variables.
            :mode: choice('set', 'add').
        '''
        start=0
        for layer in self.layers:
            stop=start+layer.num_variables
            layer.set_variables(v[start:stop], mode=mode)
            start=stop

def check_shape(pos):
    '''Check the shape of layer's method.'''
    def real_decorator(f):
        def wrapper(*args, **kwargs):
            #x, y, dy
            argss=list(args)
            if len(argss)==1:
                argss.append(kwargs.get('x'))
            if len(argss)==2:
                argss.append(kwargs.get('y'))
            if len(argss)==3:
                argss.append(kwargs.get('dy'))
            for p in pos:
                if p > 0:
                    args[0]._check_input(argss[p])
                else:
                    args[0]._check_output(argss[-p])
            return f(*args, **kwargs)
        return wrapper
    return real_decorator

def check_numdiff(layer, x, num_check=10, eta=None, tol=1e-1):
    '''Random Numerical Differential check.'''
    x=np.asfortranarray(x, dtype=layer.dtype)
    y=layer.forward(x)
    dy0=np.ones_like(y)
    dv, dx=layer.backward(x,y,dy=dy0)
    dx_=dx.ravel(order='F')
    if eta is None:
        eta=0.003+0.004j if np.iscomplexobj(dv) else 0.005

    res=True
    #check dy/dx
    for i in range(num_check):
        #change variables at random position.
        pos=np.random.randint(0,x.size)
        x_new=x.copy(order='F')
        x_new.ravel(order='F')[pos]+=eta
        y1=layer.forward(x_new)
        #print 'XBP Diff = %s, Num Diff = %s'%(dx_[pos], np.sum(y1-y)/eta)
        diff=abs(dx_[pos]-np.sum(y1-y)/eta)
        if diff/max(1,abs(dx_[pos]))>tol:
            print 'Num Diff Test Fail! @x_[%s] = %s'%(pos, x.ravel()[pos])
            res=False

    if layer.num_variables==0:
        return res

    #check dy/dw
    var0 = layer.get_variables()
    for i in range(num_check):
        #change variables at random position.
        var=var0.copy()
        pos=np.random.randint(0,var.size)
        var[pos]+=eta
        layer.set_variables(var)
        y1=layer.forward(x)
        #print 'WBP Diff = %s, Num Diff = %s'%(dv[pos], np.sum(y1-y)/eta)
        diff=abs(dv[pos]-np.sum(y1-y)/eta)
        if diff/max(1, abs(dv[pos]))>tol:
            print 'Num Diff Test Fail! @var[%s] = %s'%(pos,var[pos])
            res=False
    return res
