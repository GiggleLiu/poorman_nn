'''
ABC of neural network.
'''

import numpy as np
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import pdb

from .utils import _connect, dtype2token, dtype_c2r

__all__=['Layer','Function', 'EXP_OVERFLOW', 'EMPTY_VAR']

'''
List of tags:
    :runtimes: list of str, runtime variables, that change during each forward, [] by default.
    :is_inplace: bool, True if the output is made by changing input inplace, False by default.
    :analytical: int,
        * 0, no
        * 1, yes (default)
        * 2, yes for float, no for complex, complex output for real output.
        * 3, yes for float, no for complex, complex output for complex input.
'''
TAG_LIST = ['runtimes', 'is_inplace', 'analytical']

EXP_OVERFLOW = 12
EMPTY_VAR = np.zeros([0], dtype='float32')

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
        self.tags = {
                'runtimes':[],
                'is_inplace': False,
                'analytical': 1,
                }
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
    def forward(self,x):
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

class Monitor(Function):
    __metaclass__ = ABCMeta

    @abstractmethod
    def monitor_forward(self, x):
        pass

    @abstractmethod
    def monitor_backward(self, xy, dy, **kwargs):
        pass

    def forward(self, x):
        self.monitor_forward(x)
        return x

    def backward(self, xy, dy, **kwargs):
        self.monitor_backward(xy, dy, **kwargs)
        return EMPTY_VAR, dy
