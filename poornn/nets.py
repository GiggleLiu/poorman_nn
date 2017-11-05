'''
ABC of neural network.
'''

import numpy as np
import pdb
import numbers
import uuid

from .checks import check_shape_forward, check_shape_backward,\
    check_shape_match
from .core import Container, Function
from . import functions
from .spconv import SPConv
from .linears import Linear
from .utils import _connect, dtype2token, dtype_r2c, dtype_c2r,\
        fsign, y_from_update_res

__all__ = ['ANN', 'ParallelNN', 'JointComplex', 'KeepSignFunc']


class ANN(Container):
    '''
    Sequential Artificial Neural network.
    '''
    _id = 0
    def __init__(self, *args, **kwargs):
        super(ANN, self).__init__(*args, **kwargs)
        self.uuid = ANN._id
        ANN._id += 1

    def __graphviz__(self, g, father=None):
        node = 'cluster-%s' % id(self)
        label = '<%s<br align="left"/><font color="#225566">\
dtype = %s</font><br align="l"/>>' % (
            self.__class__.__name__, self.dtype)

        # as a container, add contents
        c = g.add_subgraph(name=node, shape='box', color='#FFCCAA',
                           label=label, labeljust='l', penwidth="5pt")

        father_ = None
        for i, layer in enumerate(self.layers):
            father_ = layer.__graphviz__(c, father=father_)
        _connect(g, father, c, self.input_shape, self.itype, pos='first')
        return c

    def __getitem__(self, name):
        if isinstance(name, numbers.Number):
            return self.layers[name]
        elif isinstance(name, str) and name in self.__layer_dict__:
            return self.__layer_dict__[name]
        else:
            raise KeyError('Get invalid key %s' % name)

    def __hash__(self):
        return hash(self.__str__())

    @property
    def input_shape(self):
        if self.num_layers == 0:
            return None
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        if self.num_layers == 0:
            return None
        return self.layers[-1].output_shape

    @property
    def itype(self):
        if self.num_layers == 0:
            return None
        return self.layers[0].itype

    @property
    def otype(self):
        if self.num_layers == 0:
            return None
        return self.layers[-1].otype

    def check_connections(self):
        layers = self.layers
        if len(layers) < 2:
            return
        for la, lb in zip(layers[:-1], layers[1:]):
            shape = check_shape_match(lb.input_shape, la.output_shape)

    def forward(self, x, data_cache=None, do_shape_check=False, **kwargs):
        '''
        Feed input to this feed forward network.

        Args:
            x (ndarray): input in 'F' order.
            data_cache (dict|None, default=None): a dict used to collect datas.
            do_shape_check (bool): check shape of data flow if True.

        Note:
            :data:`data_cache` should be pass to this method if you are about \
to call a subsequent :meth:`backward` method, \
because backward need :data:`data_cache`.

            :data:`self.uuid` is used as the key to store \
run-time output of layers in this network.
            :data:`data_cache[self.uuid]` is a list with contents \
outputs in each layers generate in this forward run.

        Returns:
            list: output in each layer.
        '''
        ys = []
        for layer in self.layers:
            if do_shape_check:
                x = check_shape_forward(layer.forward)(
                    layer, x, data_cache=data_cache, **kwargs)
            else:
                x = layer.forward(x, data_cache=data_cache, **kwargs)
            ys.append(x)
            if isinstance(x, list):
                x = x[-1]
        if data_cache is not None:
            data_cache[self.uuid] = ys
        return x

    def backward(self, xy, dy=None, data_cache=None,
                 do_shape_check=False, **kwargs):
        '''
        Compute gradients.

        Args:
            xy (tuple): input and output
            dy (ndarray): gradient of output defined as \
:math:`\partial J/\partial y`.
            data_cache (dict): a dict with collected datas.
            do_shape_check (bool): check shape of data flow if True.

        Returns:
            list: gradients for vairables in layers.
        '''
        dvs = []
        x_broken = False
        x, y = xy
        if dy is None:
            dy = np.ones(self.output_shape, dtype=self.otype)
        key = self.uuid
        if data_cache is None or key not in data_cache:
            raise TypeError('Can not find cached ys! get %s' % data_cache)
        else:
            xy = [x] + data_cache[key]
        for i in range(1, len(xy)):
            x, y = xy[-i - 1], xy[-i]
            layer = self.layers[-i]
            if do_shape_check:
                dv, dy = check_shape_backward(layer.backward)(
                    layer, [x, y], dy, data_cache=data_cache, **kwargs)
            else:
                dv, dy = layer.backward([x, y], dy, data_cache=data_cache, **kwargs)
            dvs.append(dv)
        return np.concatenate(dvs[::-1]), dy

    def add_layer(self, cls, label=None, **kwargs):
        '''
        Add a new layer, comparing with :meth:`self.layers.append`

            * :attr:`input_shape` of new layer is infered from \
:attr:`output_shape` of last layer.
            * :attr:`itype` of new layer is infered from \
:attr:`otype` of last layer.

        Args:
            cls (class): create a layer instance, take input_shape and \
itype as first and second parameters.
            label (str|None, default=None): label to index this layer, \
leave `None` if indexing is not needed.
            **kwargs: keyword arguments used by :meth:`cls.__init__`, \
excluding :attr:`input_shape` and :attr:`itype`.

        Note:
            if :attr:`num_layers` is 0, this function will raise an error,
            because it fails to infer :attr:`input_shape` and :attr:`itype`.

        Returns:
            Layer: newly generated object.
        '''
        if len(self.layers) == 0:
            raise AttributeError(
                'Please make sure this network is non-empty \
before using @add_layer.')
        else:
            input_shape, itype = self.layers[-1].output_shape,\
                self.layers[-1].otype
        obj = cls(input_shape=input_shape, itype=itype, **
                  kwargs) if not issubclass(cls, Container) else cls(**kwargs)
        self.layers.append(obj)
        if label is not None:
            self.__layer_dict__[label] = obj
        return obj


class ParallelNN(Container):
    '''
    Parallel Artificial Neural network.

    Args:
        axis (int, default=0): specify the additional axis \
on which outputs are packed.

    Attributes:
        axis (int): specify the additional axis on which outputs are packed.
    '''

    def __init__(self, axis=0, layers=None, labels=None):
        super(ParallelNN, self).__init__(layers=layers, labels=labels)
        self.axis = axis

    def __graphviz__(self, g, father=None):
        node = 'cluster-%s' % id(self)
        label = '<%s<br align="left"/><font color="#225566">\
dtype = %s</font><br align="l"/>>' % (
            self.__class__.__name__, self.dtype)

        # as a container, add contents
        c = g.add_subgraph(name=node, shape='box', color='#AACCFF',
                           label=label, labeljust='l', penwidth="5pt")

        for i, layer in enumerate(self.layers):
            father_ = layer.__graphviz__(c, father=None)
        _connect(g, father, c, self.input_shape, self.itype, pos='mid')
        return c

    def __update__(self, locs, dx, xy0, **kwargs):
        x0, y0 = dxy0
        ys = []
        add_axis = (slice(None),) * self.axis + (None,)
        for ilayer, layer in enumerate(self.layers):
            res, info = layer.update(locs, dx, (x0,y0.take(ilayer,axis=self.axis)), **kwargs)
            y = y_from_update_res(res, info)
            ys.append(y[add_axis])
        y = np.concatenate(ys, axis=self.axis)
        return y, 0

    @property
    def itype(self):
        if self.num_layers == 0:
            return None
        return np.find_common_type([layer.itype for layer in self.layers],
                                   ()).name

    @property
    def otype(self):
        if self.num_layers == 0:
            return None
        return np.find_common_type([layer.otype for layer in self.layers],
                                   ()).name

    @property
    def input_shape(self):
        if self.num_layers == 0:
            return None
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        if self.num_layers == 0:
            return None
        output_shape = self.layers[0].output_shape
        return output_shape[:self.axis] + (self.num_layers,) +\
            output_shape[self.axis:]

    def check_connections(self):
        layers = self.layers
        # check connections, same input and same output.
        if len(layers) < 2:
            return
        input_shape = layers[0].input_shape
        for la in layers[1:]:
            input_shape = check_shape_match(input_shape, la.input_shape)
            output_shape = check_shape_match(output_shape, la.output_shape)
        for la in layers:
            la.input_shape = input_shape
            la.output_shape = output_shape

    def forward(self, x, do_shape_check=False, **kwargs):
        '''
        Feed input, it will generate a new axis,\
and storge the outputs of layers parallel along this axis.

        Args:
            x (ndarray): input in 'F' order.
            do_shape_check (bool): check shape of data flow if True.

        Returns:
            ndarray: output,
        '''
        ys = []
        add_axis = (slice(None),) * self.axis + (None,)
        for layer in self.layers:
            if do_shape_check:
                y = check_shape_forward(layer.forward)(layer, x, **kwargs)
            else:
                y = layer.forward(x, **kwargs)
            ys.append(y[add_axis])
        y = np.concatenate(ys, axis=self.axis)
        return y

    def backward(self, xy, dy=np.array(1), do_shape_check=False, **kwargs):
        '''
        Compute gradients.

        Args:
            xy (tuple): input and output
            dy (ndarray): gradient of output defined as \
:math:`\partial J/\partial y`.
            do_shape_check (bool): check shape of data flow if True.

        Returns:
            list: gradients for vairables in layers.
        '''
        x, y = xy
        dvs = []
        dx = 0
        for i, layer in enumerate(self.layers):
            yi, dyi = y.take(i, axis=self.axis), dy.take(i, axis=self.axis)
            if do_shape_check:
                dv, dxi = check_shape_backward(
                    layer.backward)(layer, [x, yi], dyi, **kwargs)
            else:
                dv, dxi = layer.backward([x, yi], dyi, **kwargs)
            dvs.append(dv)
            dx += dxi
        return np.concatenate(dvs), dx

    def add_layer(self, cls, **kwargs):
        '''
        add a new layer, comparing with :meth:`self.layers.append`

            * `input_shape` of new layer is infered from \
`input_shape` of first layer.
            * `itype` of new layer is infered from `itype` of first layer.
            * `otype` of new layer is infered from `otype` of first layer.

        Args:
            cls (class): create a layer instance, take \
input_shape and itype as first and second parameters.
            **kwargs: keyword arguments used by `cls.__init__`, \
excluding `input_shape` and `itype`.

        Note:
            if `self.num_layers` is 0, this function will raise an error,
            because it fails to infer `input_shape`, `itype` and `otype`.

        Returns:
            Layer: newly generated object.
        '''
        if len(self.layers) == 0:
            raise AttributeError(
                'Please make sure this network is non-empty \
before using @add_layer.')
        obj = cls(input_shape=self.input_shape, itype=self.itype,
                  otype=self.otype, **kwargs)\
            if not issubclass(cls, Container) else cls(**kwargs)
        if self.num_layers > 0:
            check_shape_match(obj.output_shape, self.layers[0].output_shape)
        self.layers.append(obj)
        return obj


class JointComplex(Container):
    '''
    Function :math:`f(z) = h(x) + ig(y)`, \
where :math:`h` and :math:`g` are real functions.
    This :class:`Container` can be used to generate complex layers, \
but its non-holomophic (`analytical` type 3).

    Args:
        real (Layer): layer for real part.
        imag (Layer): layer for imaginary part.
    '''

    def __init__(self, real, imag):
        layers = [real, imag]
        super(JointComplex, self).__init__(layers, labels=None)

    @property
    def real(self):
        '''the real part layer.'''
        return self.layers[0]

    @property
    def imag(self):
        '''the imaginary part layer.'''
        return self.layers[1]

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
        tags = super(JointComplex, self).tags
        tags['analytical'] = 3
        return tags

    def check_connections(self):
        lr, li = self.layers
        check_shape_match(lr.input_shape, li.input_shape)
        check_shape_match(lr.output_shape, li.output_shape)
        if li.itype != lr.itype or li.otype != lr.otype:
            raise TypeError(
                'Layers in JointComplex container can not use \
different data types interfaces.')
        if lr.itype[:5] != 'float' or lr.otype[:5] != 'float':
            raise TypeError('Layers in JointComplex container \
should take float64 or float32 data \
types, but get (%s, %s)' % (lr.itype, lr.otype))

    def forward(self, x, **kwargs):
        h, g = self.layers
        return h.forward(x.real, **kwargs) + 1j * g.forward(x.imag, **kwargs)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        h, g = self.layers
        dvr, dxr = h.backward((x.real, y.real), dy.real, **kwargs)
        dvi, dxi = g.backward((x.imag, y.imag), dy.imag, **kwargs)
        return np.concatenate([dvr, -dvi]), dxr + 1j * dxi


class KeepSignFunc(Container):
    '''
    Function :math:`f(z) = h(|z|)\sign(z)`, where :math:`h` is a real function.
    This :class:`Container` inherit sign from input, \
so it must have same input and ouput dimension.
    It can also be used to generate complex layers, \
but its non-holomophic (`analytical` type 3).

    Args:
        is_real (bool, default=False): input is real if True.

    Attributes:
        is_real (bool): input is real if True.
    '''

    def __init__(self, h, is_real=False):
        from .checks import check_shape_match
        layers = [h]
        self.is_real = is_real
        check_shape_match(h.input_shape, h.output_shape)
        super(KeepSignFunc, self).__init__(layers, labels=None)

    @property
    def h(self):
        '''layer applied on amplitude.'''
        return self.layers[0]

    @property
    def itype(self):
        itype = self.layers[0].itype
        return dtype_r2c(itype) if not self.is_real else itype

    @property
    def otype(self):
        otype = self.layers[0].otype
        return dtype_r2c(otype) if not self.is_real else otype

    @property
    def input_shape(self): return self.layers[0].input_shape

    @property
    def output_shape(self): return self.layers[0].output_shape

    @property
    def tags(self):
        tags = super(KeepSignFunc, self).tags
        tags['analytical'] = 3
        return tags

    def check_connections(self):
        l = self.layers[0]
        if l.itype[:5] != 'float' or l.otype[:5] != 'float' \
                or l.dtype[:5] != 'float':
            raise TypeError('Layers in JointComplex container \
should take float64 or float32 data \
types, but get itype = %s, otype = %s, \
dtype = %s)' % (l.itype, l.otype, l.dtype))

    def forward(self, x, **kwargs):
        h, = self.layers
        return h.forward(np.abs(x), **kwargs) * fsign(x)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        h, = self.layers
        absx = np.abs(x)
        sx = fsign(x)
        sxc = sx.conj()
        hy = (sxc * y).real
        sdy = dy * sx

        dw0, dx0 = h.backward((absx, hy), sdy.real, **kwargs)
        # sdy.imag can be non-zeros.
        return dw0, dx0 * sxc + hy / np.maximum(1e-15, absx)\
            * sxc * 1j * sdy.imag
