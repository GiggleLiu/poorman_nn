'''
ABC of neural network.
'''

import numpy as np
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import pdb

__all__=['Layer','Function', 'Tags', 'EXP_OVERFLOW', 'EMPTY_VAR']

'''
Attributes for Tags:
    :runtimes: list of str, runtime variables, that change during each forward.
    :is_inplace: bool, True if the output is made by changing input inplace.
'''

Tags = namedtuple('Tags',('runtimes', 'is_inplace'))
EXP_OVERFLOW = 12
EMPTY_VAR = lambda dtype: np.zeros([0], dtype=dtype)

class Layer(object):
    '''
    A single layer in Neural Network.

    Attributes:
        :input_shape: tuple,
        :output_shape: tuple,
        :dtype: str, input data type.
        :otype: str, output data type.
        :tags: named tuple, runtime variables, is inplace(change input) function or not.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, input_shape, output_shape, dtype, otype=None, tags=Tags(runtimes = [], is_inplace = False)):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dtype = dtype
        self.otype = otype or dtype
        self.tags = tags

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s>: %s -> %s'%(self.__class__.__name__,self.input_shape,self.output_shape)

    def set_runtime_vars(self, var_dict={}):
        '''
        Set runtime variables for layers.
        '''
        for key in self.tags.runtimes:
            if not var_dict.has_key(key):
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
    def __call__(self,x):
        return self.forward(x)

    def get_variables(self):
        return EMPTY_VAR(self.dtype)

    def set_variables(self,*args,**kwargs):
        pass

    @property
    def num_variables(self):
        return 0


