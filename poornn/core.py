'''
ABC of neural network.
'''

import numpy as np
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import pdb

from .utils import _connect, dtype2token, dtype_c2r

__all__=['Layer','Function', 'ParamFunction', 'Monitor', 'EXP_OVERFLOW', 'EMPTY_VAR', 'AnalyticityError', 'DEFAULT_TAGS']

'''
List of tags:
    :runtimes: list of str, runtime variables, that change during each forward, [] by default.
    :is_inplace: bool, True if the output is made by changing input inplace, False by default.
    :analytical: int,
        * 1, yes (default)
        * 2, yes for float, no for complex, complex output for real output.
        * 3, yes for float, no for complex, complex output for complex input.
        * 4, no
'''
TAG_LIST = ['runtimes', 'is_inplace', 'analytical']

EXP_OVERFLOW = 12
EMPTY_VAR = np.zeros([0], dtype='float32')
DEFAULT_TAGS = {
            'runtimes':[],
            'is_inplace': False,
            'analytical': 1,
        }

class Layer(object):
    '''
    A single layer in Neural Network.

    Attributes:
        :input_shape: tuple,
        :output_shape: tuple,
        :itype: str, input data type.
        :otype: str, output data type.
        :tags: dict, runtime variables, is inplace(change input) function or not.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, input_shape, output_shape, itype, dtype=None, otype=None, tags=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.itype = itype
        if dtype is None: dtype = itype
        self.dtype=dtype

        # set tags
        self.tags = dict(DEFAULT_TAGS)
        if tags is not None:
            for k, v in tags.items():
                if k not in TAG_LIST:
                    print('You have used a user defined tag %s'%k)
                self.tags[k] = v

        if otype is None:
            if self.tags['analytical']==2:
                otype = dtype_c2r(itype) if itype[:7]=='complex' else itype
            else:
                otype = itype
        self.otype=otype

    def __str__(self, offset=0):
        s = ' '*offset+self.__repr__()
        if hasattr(self,'__display_attrs__'):
            for attr in self.__display_attrs__:
                s+='\n'+' '*offset+'  - %s = %s'%(attr, getattr(self,attr))
        return s

    def __repr__(self, offset=0):
        return '<%s|%s>: %s|%s -> %s|%s'%(self.__class__.__name__,dtype2token(self.dtype),self.input_shape,
                dtype2token(self.itype),self.output_shape,dtype2token(self.otype))

    def __graphviz__(self, g, father=None):
        node_token = '%s'%id(self)
        label = '<%s<br/>'%(self.__class__.__name__)
        attrs = ['itype']
        if hasattr(self, '__display_attrs__'):
            attrs.extend(self.__display_attrs__)
        for attr in attrs:
            label+='<font color="#225566" point-size="10px"> %s = %s</font><br align="left"/>'%(attr, getattr(self,attr))
        label+='>'
        g.add_node(node_token, label=label, shape='box')
        node = g.get_node(node_token)
        _connect(g, father, node, self.input_shape, self.itype)
        return node

    def set_runtime_vars(self, var_dict={}):
        '''
        Set runtime variables for layers.
        '''
        for key in self.tags['runtimes']:
            if not key in var_dict:
                raise KeyError('Variable `%s` not found, which is required by %s'%(key, self))
            self.__setattr__(key, var_dict[key])

    @abstractmethod
    def forward(self,x, **kwargs):
        '''
        Forward propagration to evaluate F(x).

        Parameters:
            :x: ndarray, input array.
            :runtime_vars: dict, runtime variables.

        Return:
            ndarray, output array y.
        '''
        pass

    @abstractmethod
    def backward(self,xy,dy,mask=(1,1)):
        '''
        Back propagation.

        Parameters:
            :xy: tuple of ndarray, input/output array.
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
    def set_variables(self, variables):
        '''
        Change current variables.

        Parameters:
            :variables: 1darray,
        '''
        pass

    @property
    @abstractmethod
    def num_variables(self):
        '''Number of variables.'''
        pass

class Function(Layer):
    '''Function layer with no variables.'''
    __metaclass__ = ABCMeta

    def __call__(self,x):
        return self.forward(x)

    def get_variables(self):
        return EMPTY_VAR

    def set_variables(self,*args,**kwargs):
        pass

    @property
    def num_variables(self):
        return 0

class ParamFunction(Layer):
    '''Function layer with params as variables and var_mask as variable mask.'''
    __metaclass__ = ABCMeta

    def __init__(self, input_shape, output_shape, itype, params, var_mask, **kwargs):
        self.params = np.atleast_1d(params)
        if var_mask is None:
            var_mask = np.ones(len(params),dtype='bool')
        else:
            var_mask = np.asarray(var_mask, dtype='bool')
        self.var_mask = np.atleast_1d(var_mask)
        if 'dtype' in kwargs:
            dtype = kwargs.pop('dtype')
            self.params = self.params.astype(dtype)
        else:
            dtype = self.params.dtype.name
        otype = kwargs.pop('otype', np.find_common_type((dtype, itype),()).name)
        super(ParamFunction,self).__init__(input_shape, output_shape, itype, dtype = dtype, otype=otype, **kwargs)

    def __call__(self,x):
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
        :layers: list,
        :do_shape_check: bool,
    '''
    __metaclass__ = ABCMeta

    def __init__(self, layers=None, labels=None):
        if layers is None: layers = []
        if labels is None: labels = []
        self.layers = layers
        self.__layer_dict__ = dict(zip(labels,layers))

        # itype, dtype, otype, input_shape and output_shape are defined as properties.

        #check connections
        self.check_connections()

    def __str__(self, offset=0):
        s = ' '*offset+self.__repr__()
        for layer in self.layers:
            s+='\n'+layer.__str__(offset=offset+4)
        return s

    def __repr__(self, offset=0):
        return '<%s|%s>: %s|%s -> %s|%s'%(self.__class__.__name__,dtype2token(self.dtype),self.input_shape,
                dtype2token(self.itype),self.output_shape,dtype2token(self.otype))

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def tags(self):
        runtimes = []
        analytical = 1
        is_inplace = False
        for i,layer in enumerate(self.layers):
            if hasattr(layer,'tags'):
                runtimes.extend(layer.tags.get('runtimes',[]))
                analytical = max(analytical,layer.tags.get('analytical',1))
                if i==0:
                    is_inplace = layer.tags.get('is_inplace',False)
        if analytical==3 and self.otype[:5] == 'float':
            analytical = 2
        if analytical==2 and self.otype[:7] == 'complex':
            analytical = 3
        return {'runtimes': runtimes,
                'analytical': analytical,
                'is_inplace': is_inplace,
                }

    @property
    @abstractmethod
    def itype(self):
        pass

    @property
    @abstractmethod
    def otype(self):
        pass

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def output_shape(self):
        pass

    @property
    def dtype(self):
        if self.num_layers==0:
            raise AttributeError('Can not infer dtype from empty network.')
        return np.find_common_type([layer.dtype for layer in self.layers],()).name

    def check_connections(self):
        pass

    def get_runtimes(self):
        '''Show requested runtime variables'''
        rd = {}
        for layer in layers:
            for key in layer.tags['runtimes']:
                value=layer.__getattribute__(key)
                if hasattr(rd, key) and (value is not rd[var]):
                    raise Exception('runtime variables conflicts %s and %s not same'%(rd[var], value))
                rd[var]=value
        return rd

    def set_runtime_vars(self, var_dict):
        '''
        Set runtime variables.
        '''
        for layer in self.layers:
            layer.set_runtime_vars(var_dict)

    def get_variables(self):
        '''Dump values to an array.'''
        return np.concatenate([layer.get_variables() for layer in self.layers])

    def set_variables(self,v):
        '''
        Load data from an array.
        
        Parameters:
            :v: 1darray, variables.
        '''
        start=0
        for layer in self.layers:
            stop=start+layer.num_variables
            layer.set_variables(np.asarray(v[start:stop]))
            start=stop

    @property
    def num_variables(self):
        return np.sum([layer.num_variables for layer in self.layers])

class Monitor(Function):
    __metaclass__ = ABCMeta

    @abstractmethod
    def monitor_forward(self, x):
        pass

    @abstractmethod
    def monitor_backward(self, xy, dy, **kwargs):
        pass

    def forward(self, x, **kwargs):
        self.monitor_forward(x)
        return x

    def backward(self, xy, dy, **kwargs):
        self.monitor_backward(xy, dy, **kwargs)
        return EMPTY_VAR, dy

class AnalyticityError(Exception):
    pass
