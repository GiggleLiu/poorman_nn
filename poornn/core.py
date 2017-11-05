'''
ABC of neural network.
'''

import numpy as np
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import pdb

from .utils import _connect, dtype2token, dtype_c2r, get_tag

__all__ = ['Layer', 'Function', 'ParamFunction', 'Monitor', 'EXP_OVERFLOW',
           'EMPTY_VAR', 'AnalyticityError', 'DEFAULT_TAGS', 'TAG_LIST']

TAG_LIST = ['runtimes', 'is_inplace', 'analytical', 'one2one']
'''
List of tags:

    * 'runtimes' (list<str>, default=[]):
        runtime variables that should be supplied during each run.
    * 'is_inplace' (bool, default=False):
        True if the output is made by changing input inplace.
    * 'analytical' (int):
        the analyticaity of a layer. A table of legal values,

        * 1, yes (default)
        * 2, yes for float, no for complex, complex output for real output.
        * 3, yes for float, no for complex, complex output for complex input.
        * 4, no
    * 'one2one' (bool):
        True if this layer performs a one-to-one mapping,
        neither inducing entanglement between data, nor chaning the order.
'''

EXP_OVERFLOW = 12
'''
exp(x>EXP_OVERFLOW) should be taken special care of in order avoid overflow.
'''

EMPTY_VAR = np.zeros([0], dtype='float32')
'''Empty variable, 1d array of dtype 'float32' and length 0.'''

DEFAULT_TAGS = {
    'runtimes': [],
    'is_inplace': False,
    'analytical': 1,
    'one2one': False,
}
'''
A layer without tags attributes will take this set of tags.

    * no runtime variables,
    * changes for flow are not inplace (otherwise it will destroy integrity of flow history).
    * analytical (for complex numbers, holomophic).
'''


class Layer(object):
    '''
    A single layer in Neural Network.

    Args:
        input_shape (tuple): input shape of this layer.
        output_shape (tuple): output_shape of this layer.
        itype (str): input data type.
        dtype (str, default=:data:`itype`): variable data type.
        otype (str, default=?): output data type, if not provided,
            it will be set to itype, unless its 'analytical' tags is 2.
        tags (dict, default=:data:`poornn.core.DEFAULT_TAGS`): tags used \
to describe this layer, refer :data:`poornn.core.TAG_LIST` for detail. \
It change tags based on template :data:`poornn.core.DEFAULT_TAGS`.

    Attributes:
        input_shape (tuple): input shape of this layer.
        output_shape (tuple): output_shape of this layer.
        itype (str): input data type.
        dtype (str): variable data type.
        otype (str): output data type.
        tags (dict): tags used to describe this layer, \
refer :data:`poornn.core.TAG_LIST` for detail.
    '''

    __metaclass__ = ABCMeta
    __display_attrs__ = []
    '''
    except :attr:`input_shape`/:attr:`output_shape` \
and :attr:`itype`/:attr:`dtype`/:attr:`otype`, \
attributes that will be displayed in print and graphviz.
    '''

    def __init__(self, input_shape, output_shape,
                 itype, dtype=None, otype=None, tags=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.itype = itype
        if dtype is None:
            dtype = itype
        self.dtype = dtype

        # set tags
        self.tags = dict(DEFAULT_TAGS)
        if tags is not None:
            for k, v in tags.items():
                if k not in TAG_LIST:
                    print('You have used a user defined tag %s' % k)
                self.tags[k] = v

        if otype is None:
            if self.tags['analytical'] == 2:
                otype = dtype_c2r(itype) if itype[:7] == 'complex' else itype
            else:
                otype = itype
        self.otype = otype

    def __str__(self, offset=0):
        s = ' ' * offset + self.__repr__()
        if hasattr(self, '__display_attrs__'):
            for attr in self.__display_attrs__:
                s += '\n' + ' ' * offset + \
                    '  - %s = %s' % (attr, getattr(self, attr))
        return s

    def __repr__(self, offset=0):
        return '<%s|%s>: %s|%s -> %s|%s' % (self.__class__.__name__,
                                            dtype2token(
                                                self.dtype), self.input_shape,
                                            dtype2token(self.itype),
                                            self.output_shape,
                                            dtype2token(self.otype))

    def __graphviz__(self, g, father=None):
        node_token = '%s' % id(self)
        label = '<%s<br/>' % (self.__class__.__name__)
        attrs = ['dtype']
        if hasattr(self, '__display_attrs__'):
            attrs.extend(self.__display_attrs__)
        for attr in attrs:
            label += '<font color="#225566" point-size="10px">\
                    %s = %s</font><br align="left"/>' % (
                attr, getattr(self, attr))
        label += '>'
        g.add_node(node_token, label=label, shape='box')
        node = g.get_node(node_token)
        _connect(g, father, node, self.input_shape, self.itype)
        return node

    def update(self, locs, dx, xy0, **kwargs):
        '''
        update forward run using existing run.

        Args:
            locs (2darray): different positions in input.
            dx (ndarray): difference amount of value in input.
            xy0 (tuple): x0 and y0 in previous run.

        Return:
            tuple|ndarray, int: data and information,
                * information == 0, data is entangled, terminating (or swich to forward), data is ndarray.
                * information == 1, not entangled, can go one update run, data is tuple.
        '''
        x0, y0 = xy0
        if hasattr(self,'__update__'):
            return self.__update__(locs, dx, xy0, **kwargs)
        else:
            if get_tag(self, 'one2one'):
                y = self.forward(x0[locs]+dx, **kwargs)
                return (locs, y-y0[locs]), 1
            else:
                raise NotImplementedError('Layer %s do not support history based update.'%self.__class__.__name__)

    def set_runtime_vars(self, var_dict={}):
        '''
        Set runtime variables for layers.

        Args:
            var_dict (dict): the runtime variables dict.
        '''
        for key in self.tags['runtimes']:
            if key not in var_dict:
                raise KeyError(
                    'Variable `%s` not found, which is required by %s' % (
                        key, self))
            self.__setattr__(key, var_dict[key])

    @abstractmethod
    def forward(self, x, **kwargs):
        '''
        forward propagration to evaluate :math:`y=f(x)`.

        Args:
            x (ndarray): input array.
            runtime_vars (dict): runtime variables.

        Returns:
            ndarray, output array y.
        '''
        pass

    @abstractmethod
    def backward(self, xy, dy, mask=(1, 1)):
        '''
        back propagation to get :math:`\\frac{\partial J(w,x)}{\partial w}` \
and :math:`\\frac{\partial J(w,x)}{\partial x}`, \
where :math:`J` and :math:`w` are cost function and variables respectively.

        Args:
            xy (tuple<ndarray>, len=2): input and output array.
            dy (ndarray): gradient of output defined as \
:math:`\partial J/\partial y`.
            mask (tuple): (do_wgrad, do_xgrad)

        Returns:
            (ndarray, ndarray), :math:`\partial J/\partial w` and \
:math:`\partial J/\partial x`.
        '''
        pass

    @abstractmethod
    def get_variables(self):
        '''
        Get current variables.

        Returns:
            1darray,
        '''
        pass

    @abstractmethod
    def set_variables(self, variables):
        '''
        Change current variables.

        Args:
            variables (1darray):
        '''
        pass

    @property
    @abstractmethod
    def num_variables(self):
        '''number of variables.'''
        pass


class Function(Layer):
    '''Function layer with no variables.'''
    __metaclass__ = ABCMeta

    def __call__(self, x):
        return self.forward(x)

    def get_variables(self):
        '''Get variables, return empty (1d but with length - 0) array.'''
        return EMPTY_VAR

    def set_variables(self, *args, **kwargs):
        '''passed.'''
        pass

    @property
    def num_variables(self):
        '''number of variables, which is fixed to 0.'''
        return 0


class ParamFunction(Layer):
    '''
    Function layer with params as variables and var_mask as variable mask.

    Args:
        params (1darray): variables used in this functions.
        var_mask (1darray<bool>, default=(True,True,...)): mask for params, \
a param is regarded as a constant if its mask is False.

    Attributes:
        params (1darray): variables used in this functions.
        var_mask (1darray<bool>): mask for params, \
a param is regarded as a constant if its mask is False.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, input_shape, output_shape, itype,
                 params, var_mask, **kwargs):
        self.params = np.atleast_1d(params)
        if var_mask is None:
            var_mask = np.ones(len(params), dtype='bool')
        else:
            var_mask = np.asarray(var_mask, dtype='bool')
        self.var_mask = np.atleast_1d(var_mask)
        if 'dtype' in kwargs:
            dtype = kwargs.pop('dtype')
            self.params = self.params.astype(dtype)
        else:
            dtype = self.params.dtype.name
        otype = kwargs.pop(
            'otype', np.find_common_type((dtype, itype), ()).name)
        super(ParamFunction, self).__init__(input_shape,
                                            output_shape, itype,
                                            dtype=dtype, otype=otype,
                                            **kwargs)

    def __call__(self, x):
        return self.forward(x)

    def get_variables(self):
        return self.params[self.var_mask]

    def set_variables(self, a):
        self.params[self.var_mask] = a

    @property
    def num_variables(self):
        return self.var_mask.sum()


class Container(Layer):
    '''
    Function layer with no variables.

    Attributes:
        layers (list<Layer>): layers.
        labels (list<str>): labels for layers, used for query.

    Attributes:
        layers (list<Layer>, default=[]): layers.
        labels (list<str>, default=[]): labels for layers, used for query.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, layers=None, labels=None):
        if layers is None:
            layers = []
        if labels is None:
            labels = []
        self.layers = layers
        self.__layer_dict__ = dict(zip(labels, layers))

        # itype, dtype, otype, input_shape and
        # output_shape are defined as properties.

        # check connections
        self.check_connections()

    def __str__(self, offset=0):
        s = ' ' * offset + self.__repr__()
        for layer in self.layers:
            s += '\n' + layer.__str__(offset=offset + 4)
        return s

    def __repr__(self, offset=0):
        if self.num_layers == 0:
            return '<%s|empty>'%self.__class__.__name__
        return '<%s|%s>: %s|%s -> %s|%s' % (self.__class__.__name__,
                                            dtype2token(
                                                self.dtype), self.input_shape,
                                            dtype2token(self.itype),
                                            self.output_shape,
                                            dtype2token(self.otype))

    @property
    def num_layers(self):
        '''number of layers.'''
        return len(self.layers)

    @property
    def tags(self):
        '''tags for this :class:`Container`, which is infered \
from :attr:`self.layers`.'''
        runtimes = []
        analytical = 1
        is_inplace = False
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'tags'):
                runtimes.extend(layer.tags.get('runtimes', []))
                analytical = max(analytical, layer.tags.get('analytical', 1))
                if i == 0:
                    is_inplace = layer.tags.get('is_inplace', False)
        if analytical == 3 and self.otype[:5] == 'float':
            analytical = 2
        if analytical == 2 and self.otype[:7] == 'complex':
            analytical = 3
        return {'runtimes': runtimes,
                'analytical': analytical,
                'is_inplace': is_inplace,
                }

    @property
    @abstractmethod
    def itype(self):
        '''input data type, which is infered from layers.'''
        pass

    @property
    @abstractmethod
    def otype(self):
        '''output data type, which is infered from layers.'''
        pass

    @property
    @abstractmethod
    def input_shape(self):
        '''input data shape, which is infered from layers.'''
        pass

    @property
    @abstractmethod
    def output_shape(self):
        '''output data shape, which is infered from layers.'''
        pass

    @property
    def dtype(self):
        '''variable data shape, which is infered from layers.'''
        if self.num_layers == 0:
            raise AttributeError('Can not infer dtype from empty network.')
        return np.find_common_type([layer.dtype for layer in self.layers],
                                   ()).name

    def check_connections(self):
        '''
        check whether connections among layers in this \
:class:`Container` are valid.
        '''
        pass

    def get_runtimes(self):
        '''show runtime variables used in this :class:`Container`.'''
        rd = {}
        for layer in layers:
            for key in layer.tags['runtimes']:
                value = layer.__getattribute__(key)
                if hasattr(rd, key) and (value is not rd[var]):
                    raise Exception(
                        'runtime variables conflicts %s \
                                and %s not same' % (rd[var], value))
                rd[var] = value
        return rd

    def set_runtime_vars(self, var_dict):
        '''
        Set runtime variables.

        Args:
            var_dict (dict): the runtime variables dict.
        '''
        for layer in self.layers:
            layer.set_runtime_vars(var_dict)

    def get_variables(self):
        '''Dump values to an array.'''
        return np.concatenate([layer.get_variables() for layer in self.layers])

    def set_variables(self, v):
        '''
        Load data from an array.

        Args:
            v (1darray): variables.
        '''
        start = 0
        for layer in self.layers:
            stop = start + layer.num_variables
            layer.set_variables(np.asarray(v[start:stop]))
            start = stop

    @property
    def num_variables(self):
        '''int: number of variables.'''
        return np.sum([layer.num_variables for layer in self.layers])


class Monitor(Function):
    '''
    A special layer used to monitor a flow, it operate on but do not \
change the flow.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def monitor_forward(self, x):
        '''
        Monitor function used in forward,

        Args:
            x (ndarray): forward data.
        '''
        pass

    @abstractmethod
    def monitor_backward(self, xy, dy, **kwargs):
        '''
        Monitor function used in backward,

        Args:
            xy (ndarray): (input, output(same as input)) data.
            dy (ndarray): gradient.
        '''
        pass

    def forward(self, x, **kwargs):
        self.monitor_forward(x)
        return x

    def backward(self, xy, dy, **kwargs):
        self.monitor_backward(xy, dy, **kwargs)
        return EMPTY_VAR, dy


class AnalyticityError(Exception):
    '''Behavior conflict with the analytical type of a layer.'''
    pass
