'''
ABC of neural network.
'''

import numpy as np
import pdb, numbers

from .checks import check_shape_forward, check_shape_backward, check_shape_match
from .core import Container, Function
from . import functions
from .spconv import SPConv
from .linears import Linear
from .utils import _connect, dtype2token, dtype_r2c, dtype_c2r, fsign

__all__=['ANN', 'ParallelNN', 'JointComplex']

class ANN(Container):
    '''
    Sequential Artificial Neural network,
    '''
    def __graphviz__(self, g, father=None):
        node = 'cluster-%s'%id(self)
        label='<%s<br align="left"/><font color="#225566">itype = %s</font><br align="l"/>>'%(self.__class__.__name__, self.itype)

        # as a container, add contents
        c = g.add_subgraph(name=node, shape='box', color='#FFCCAA',
                label=label, labeljust='l', penwidth="5pt")

        father_ = None
        for i,layer in enumerate(self.layers):
            father_ = layer.__graphviz__(c, father=father_)
        _connect(g, father, c, self.input_shape, self.itype, pos='first')
        return c

    def __getitem__(self, name):
        if isinstance(name,numbers.Number):
            return self.layers[name]
        elif isinstance(name,str) and name in self.__layer_dict__:
            return self.__layer_dict__[name]
        else:
            raise KeyError('Get invalid key %s'%name)

    @property
    def input_shape(self):
        if self.num_layers==0:
            return None
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        if self.num_layers==0:
            return None
        return self.layers[-1].output_shape

    @property
    def itype(self):
        if self.num_layers==0:
            return None
        return self.layers[0].itype

    @property
    def otype(self):
        if self.num_layers==0:
            return None
        return self.layers[-1].otype

    def check_connections(self):
        layers = self.layers
        if len(layers)<2: return
        for la,lb in zip(layers[:-1], layers[1:]):
            shape = check_shape_match(lb.input_shape, la.output_shape)

    def forward(self, x, return_info=True, do_shape_check=False):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), in 'F' order.

        Return:
            list, output in each layer.
        '''
        xy=[x]
        for layer in self.layers:
            if do_shape_check:
                x=check_shape_forward(layer.forward)(layer, x)
            else:
                x=layer.forward(x)
            xy.append(x)
        return xy

    def backward(self, xy, dy=np.array(1), do_shape_check=False):
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
            if do_shape_check:
                dv, dy=check_shape_backward(layer.backward)(layer, [x, y], dy)
            else:
                dv, dy=layer.backward([x, y], dy)
            dvs.append(dv)
        return np.concatenate(dvs[::-1]), dy

    def add_layer(self, cls, label=None, **kwargs):
        '''
        Add a new layer. *args and **kwargs specifies parameters excluding `input_shape` and `itype`.
        input_shape inherit the output_shape of last layer, and itype inherit otype of last layer.

        Parameters:
            :cls: class, create a layer instance, take input_shape and itype as first and second parameters.
            :label: str, label to index this layer.
        '''
        if len(self.layers)==0:
            raise AttributeError('Please make sure this network is non-empty before using @add_layer.')
        else:
            input_shape, itype = self.layers[-1].output_shape, self.layers[-1].otype
        obj=cls(input_shape=input_shape, itype=itype, **kwargs) if not issubclass(cls,Container) else cls(**kwargs)
        self.layers.append(obj)
        if label is not None:
            self.__layer_dict__[label] = obj
        return obj


class ParallelNN(Container):
    '''
    Parallel Artificial Neural network,

    Attributes:
        :itype: str, the most `advanced` data type used, like 'complex128' if both 'complex128' and 'float32' are used.
        :axis: int, specify the additional axis.
        :layers: list,
    '''
    def __init__(self, axis=0, layers=None, labels=None):
        super(ParallelNN,self).__init__(layers=layers, labels=labels)
        self.axis = axis

    def __graphviz__(self, g, father=None):
        node = 'cluster-%s'%id(self)
        label='<%s<br align="left"/><font color="#225566">itype = %s</font><br align="l"/>>'%(self.__class__.__name__, self.itype)

        # as a container, add contents
        c = g.add_subgraph(name=node, shape='box', color='#AACCFF',
                label=label, labeljust='l', penwidth="5pt")

        for i,layer in enumerate(self.layers):
            father_ = layer.__graphviz__(c, father=None)
        _connect(g, father, c, self.input_shape, self.itype, pos='mid')
        return c

    @property
    def itype(self):
        if self.num_layers==0:
            return None
        return np.find_common_type([layer.itype for layer in self.layers],()).name

    @property
    def otype(self):
        if self.num_layers==0:
            return None
        return np.find_common_type([layer.otype for layer in self.layers],()).name

    @property
    def input_shape(self):
        if self.num_layers==0:
            return None
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        if self.num_layers==0:
            return None
        output_shape = self.layers[0].output_shape
        return output_shape[:self.axis]+(self.num_layers,)+output_shape[self.axis:]

    def check_connections(self):
        layers = self.layers
        #check connections, same input and same output.
        if len(layers)<2: return
        input_shape = layers[0].input_shape
        for la in layers[1:]:
            input_shape = check_shape_match(input_shape, la.input_shape)
            output_shape = check_shape_match(output_shape, la.output_shape)
        for la in layers:
            la.input_shape = input_shape
            la.output_shape = output_shape

    def forward(self, x, return_info=False, do_shape_check=False):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), in 'F' order.

        Return:
            ndarray, output,
            (ndarray, list) if return_info, (output, outputs in each layer).
        '''
        ys = []
        for layer in self.layers:
            if do_shape_check:
                y=check_shape_forward(layer.forward)(layer, x)
            else:
                y=layer.forward(x)
            ys.append(y[(slice(None),)*self.axis+(None,)])
        y = np.concatenate(ys,axis=self.axis)
        if return_info:
            return y, {}
        else:
            return y

    def backward(self, xy, dy=np.array(1), do_shape_check=False):
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
            if do_shape_check:
                dv, dxi=check_shape_backward(layer.backward)(layer, [x, yi], dyi)
            else:
                dv, dxi=layer.backward([x, yi], dyi )
            dvs.append(dv)
            dx += dxi
        return np.concatenate(dvs), dx

    def add_layer(self, cls, **kwargs):
        '''
        Add a new layer. *args and **kwargs specifies parameters excluding `input_shape` and `itype`.
        input_shape inherit the output_shape of last layer, and itype inherit otype of last layer.
        '''
        if len(self.layers)==0:
            raise AttributeError('Please make sure this network is non-empty before using @add_layer.')
        obj=cls(input_shape=self.input_shape, itype=self.itype, otype=self.otype, **kwargs)\
                if not issubclass(cls,Container) else cls(**kwargs)
        if self.num_layers>0:
            check_shape_match(obj.output_shape, self.layers[0].output_shape)
        self.layers.append(obj)
        return obj

class JointComplex(Container):
    '''
    Function f(z) = h(x) + 1j*g(y), h and g are real functions.
    '''
    def __init__(self, real, imag, labels=None):
        layers = [real, imag]
        super(JointComplex, self).__init__(layers, labels)

    @property
    def real(self):return self.layers[0]
    @property
    def imag(self): return self.layers[1]
    @property
    def itype(self): return dtype_r2c(self.layers[0].itype)
    @property
    def otype(self): return dtype_r2c(self.layers[0].otype)
    @property
    def input_shape(self): return self.layers[0].input_shape
    @property
    def output_shape(self): return self.layers[0].output_shape

    @property
    def tags(self):
        tags = super(JointComplex,self).tags
        tags['analytical'] = 3
        return tags

    def check_connections(self):
        lr, li = self.layers
        check_shape_match(lr.input_shape,li.input_shape)
        check_shape_match(lr.output_shape,li.output_shape)
        if li.itype != lr.itype or li.otype != lr.otype:
            raise TypeError('Layers in JointComplex container can not use different data types interfaces.')
        if lr.itype[:5]!='float' or lr.otype[:5]!='float':
            raise TypeError('Layers in JointComplex container should take float64 or float32 data\
                    types, but get (%s, %s)'%(lr.itype,lr.otype))

    def forward(self, x, **kwargs):
        h, g = self.layers
        return h.forward(x.real,**kwargs)+1j*g.forward(x.imag,**kwargs)

    def backward(self,xy,dy,**kwargs):
        x,y = xy
        h, g = self.layers
        dvr, dxr = h.backward((x.real,y.real),dy.real,**kwargs)
        dvi, dxi = g.backward((x.imag,y.imag),dy.imag,**kwargs)
        return np.concatenate([dvr,-dvi]), dxr+1j*dxi

class KeepSignFunc(Container):
    '''
    Function f(z) = h(|z|)*sign(z), h is a real function.
    '''
    def __init__(self, h):
        layers = [h]
        super(KeepSignFunc, self).__init__(layers, labels=None)

    @property
    def h(self): return self.layers[0]
    @property
    def itype(self): return dtype_r2c(self.layers[0].itype)
    @property
    def otype(self): return dtype_r2c(self.layers[0].otype)
    @property
    def input_shape(self): return self.layers[0].input_shape
    @property
    def output_shape(self): return self.layers[0].output_shape

    @property
    def tags(self):
        tags = super(KeepSignFunc,self).tags
        tags['analytical'] = 3
        return tags

    def check_connections(self):
        l=self.layers[0]
        if l.itype[:5]!='float' or l.otype[:5]!='float' or l.dtype[:5]!='float':
            raise TypeError('Layers in JointComplex container should take float64 or float32 data\
                    types, but get itype = %s, otype = %s, dtype = %s)'%(l.itype,l.otype,l.dtype))

    def forward(self, x, **kwargs):
        h, = self.layers
        return h.forward(np.abs(x),**kwargs)*fsign(x)

    def backward(self,xy,dy,**kwargs):
        x, y = xy
        h, = self.layers
        absx = np.abs(x)
        sx = fsign(x)
        sxc = sx.conj()
        hy = (sxc*y).real
        sdy = dy*sx

        dw0, dx0 = h.backward((absx,hy),sdy.real)
        return dw0, dx0*sxc+hy/np.maximum(1e-15,absx)*sxc*1j*sdy.imag  #sdy.imag can be non-zeros.
