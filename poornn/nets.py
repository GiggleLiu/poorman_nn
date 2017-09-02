'''
ABC of neural network.
'''

import numpy as np
import pdb

from .checks import check_shape_forward, check_shape_backward
from .core import Layer, Function
from . import functions
from .spconv import SPConv
from .linears import Linear
from .utils import _connect

__all__=['ANN', 'ParallelNN']

class ANN(Layer):
    '''
    Sequential Artificial Neural network, is a special Layer.

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
        self.tags = None

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

    def __graphviz__(self, g, father=None):
        node = 'cluster-%s'%id(self)
        label='<%s<br align="left"/><font color="#225566">dtype = %s</font><br align="l"/>>'%(self.__class__.__name__, self.dtype)

        # as a container, add contents
        c = g.add_subgraph(name=node, shape='box', color='#FFCCAA',
                label=label, labeljust='l', penwidth="5pt")

        father_ = None
        for i,layer in enumerate(self.layers):
            father_ = layer.__graphviz__(c, father=father_)
        _connect(g, father, c, self.input_shape, self.dtype, pos='first')
        return c

    @property
    def input_shape(self):
        if self.num_layers==0:
            raise AttributeError('Can not infer input_shape from empty network.')
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        if self.num_layers==0:
            raise AttributeError('Can not infer output_shape from empty network.')
        return self.layers[-1].output_shape

    @property
    def otype(self):
        if self.num_layers==0:
            raise AttributeError('Can not infer otype from empty network.')
        return self.layers[-1].otype

    @property
    def num_layers(self):
        return len(self.layers)

    def set_runtime_vars(self, var_dict):
        '''
        Set runtime variables.
        '''
        for layer in self.layers:
            layer.set_runtime_vars(var_dict)

    def forward(self, x, return_info=True):
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
        for i in range(1,len(xy)):
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
        return obj


class ParallelNN(Layer):
    '''
    Parallel Artificial Neural network, is a special Layer.

    Attributes:
        :dtype: str, the most `advanced` data type used, like 'complex128' if both 'complex128' and 'float32' are used.
        :axis: int, specify the additional axis.
        :layers: list,
        :do_shape_check: bool,
    '''
    def __init__(self, input_shape, output_shape, dtype, axis=0, layers=None, otype=None, do_shape_check=False, tags=None):
        super(ParallelNN,self).__init__(input_shape, output_shape, dtype, otype=otype, tags=tags)
        if layers is None: layers = []
        self.layers = layers
        self.do_shape_check = do_shape_check
        self.axis = axis

        #check connections, same input and same output.
        if len(layers)<2: return
        for la in layers:
            if la.input_shape==None:
                la.input_shape=self.input_shape
            if la.output_shape is None:
                la.output_shape=self.output_shape1
            #check input and output shape
            for sa, sb in zip(la.input_shape+la.output_shape, self.input_shape+self.output_shape1):
                if sa!=sb and sa!=-1 and sb!=-1:
                    raise Exception('Shape for layers %s mismatch!'%la)

    def __str__(self):
        s='<%s>, layers ='%self.__class__.__name__
        for layer in self.layers:
            s+='\n  '+layer.__str__()
        return s

    def __graphviz__(self, g, father=None):
        node = 'cluster-%s'%id(self)
        label='<%s<br align="left"/><font color="#225566">dtype = %s</font><br align="l"/>>'%(self.__class__.__name__, self.dtype)

        # as a container, add contents
        c = g.add_subgraph(name=node, shape='box', color='#AACCFF',
                label=label, labeljust='l', penwidth="5pt")

        for i,layer in enumerate(self.layers):
            father_ = layer.__graphviz__(c, father=None)
        _connect(g, father, c, self.input_shape, self.dtype, pos='mid')
        return c

    @property
    def num_layers(self):
        return len(self.layers)

    def set_runtime_vars(self, var_dict):
        '''
        Set runtime variables.
        '''
        for layer in self.layers:
            layer.set_runtime_vars(var_dict)

    def forward(self, x, return_info=False):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), in 'F' order.

        Return:
            ndarray, output,
            (ndarray, list) if return_info, (output, outputs in each layer).
        '''
        ys = []
        for layer in self.layers:
            if self.do_shape_check:
                y=check_shape_forward(layer.forward)(layer, x)
            else:
                y=layer.forward(x)
            ys.append(y[(slice(None),)*self.axis+(None,)])
        y = np.concatenate(ys,axis=self.axis)
        if return_info:
            return y, {}
        else:
            return y

    def backward(self, xy, dy=np.array(1)):
        '''
        Compute gradients.

        Parameters:
            :xy: list, output in each layer.

        Return:
            list, gradients for vairables in layers.
        '''
        x, y = xy
        dvs=[]
        dx = 0
        for i, layer in enumerate(self.layers):
            yi, dyi = y.take(i,axis=self.axis), dy.take(i,axis=self.axis)
            if self.do_shape_check:
                dv, dxi=check_shape_backward(layer.backward)(layer, [x, yi], dyi)
            else:
                dv, dxi=layer.backward([x, yi], dyi )
            dvs.append(dv)
            dx += dxi
        return np.concatenate(dvs), dx

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

    @property
    def output_shape1(self):
        return self.output_shape[:self.axis]+self.output_shape[self.axis+1:]

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
        obj=cls(input_shape=self.input_shape, output_shape=self.output_shape1, dtype=self.dtype, otype=self.otype, **kwargs)
        self.layers.append(obj)
        return obj
