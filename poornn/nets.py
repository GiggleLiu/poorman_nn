'''
ABC of neural network.
'''

import numpy as np
import pdb

from checks import check_shape_forward, check_shape_backward
from core import Layer, Function
import functions
from spconv import SPConv
from linears import Linear

__all__=['ANN']

class ANN(object):
    '''
    Artificial Neural network state.

    Attributes:
        :dtype: str, the most `advanced` data type used, like 'complex128' if both 'complex128' and 'float32' are used.
        :layers: list,
        :do_shape_check: bool,
    '''
    def __init__(self, dtype, layers=None, do_shape_check=False):
        if layers is None: layers = []
        self.layers = layers
        self.do_shape_check = do_shape_check
        self.dtype = dtype

        #check connections
        if len(layers)<2: return
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

    def __str__(self):
        s='<%s>, layers ='%self.__class__.__name__
        for layer in self.layers:
            s+='\n  '+layer.__str__()
        return s

    @property
    def input_shape(self):
        if self.num_layers==0:
            raise AttributeError('Can not infer input_shape from empty network.')
        return self.layers[0].input_shape

    @property
    def num_layers(self):
        return len(self.layers)

    def set_runtime_vars(self, var_dict):
        '''
        Set runtime variables.
        '''
        for layer in self.layers:
            layer.set_runtime_vars(var_dict)

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), in 'F' order.

        Return:
            list, output in each layer.
        '''
        xy=[x]
        for layer in self.layers:
            if self.do_shape_check:
                x=check_shape_forward(layer.forward)(layer, x)
            else:
                x=layer.forward(x)
            xy.append(x)
        return xy

    def backward(self, xy, dy=np.array(1)):
        '''
        Compute gradients.

        Parameters:
            :xy: list, output in each layer.

        Return:
            list, gradients for vairables in layers.
        '''
        dvs=[]
        x_broken = False
        for i in xrange(1,len(xy)):
            x, y = xy[-i-1], xy[-i]
            layer = self.layers[-i]
            if self.do_shape_check:
                dv, dy=check_shape_backward(layer.backward)(layer, [x, y], dy)
            else:
                dv, dy=layer.backward([x, y], dy)
            dvs.append(dv)
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
            layer.set_variables(np.asarray(v[start:stop],dtype=layer.dtype), mode=mode)
            start=stop

    @property
    def num_variables(self):
        return np.sum([layer.num_variables for layer in self.layers])

    def get_runtimes(self):
        '''Show requested runtime variables'''
        rd = {}
        for layer in layers:
            for key in layer.tags.runtimes:
                value=layer.__getattribute__(key)
                if hasattr(rd, key) and (value is not rd[var]):
                    raise Exception('runtime variables conflicts %s and %s not same'%(rd[var], value))
                rd[var]=value
        return rd

    def add_layer(self, cls, **kwargs):
        '''
        Add a new layer. *args and **kwargs specifies parameters excluding `input_shape` and `dtype`.
        input_shape inherit the output_shape of last layer, and dtype inherit dtype of last layer(network with mixed dtype?).
        '''
        if len(self.layers)==0:
            raise AttributeError('Please make sure this network is non-empty before using @add_layer.')
        else:
            input_shape, dtype = self.layers[-1].output_shape, self.layers[-1].otype
        obj=cls(input_shape=input_shape, dtype=dtype, **kwargs)
        self.layers.append(obj)
