'''
Linear Layer.
'''

import numpy as np

from core import Layer
from conv

class L_Tensor(Layer):
    '''Tensor Layer.'''
    def __init__(self,tensor,einsum_tokens):
        self.tensor=np.asarray(tensor)
        if len(einsum_tokens)!=3:
            raise ValueError('einsum_tokens should be a len-3 list!')
        if len(einsum_tokens[1])!=self.tensor.ndim:
            raise ValueError('einsum_tokens dimension error!')

    def forward(self,x):
        dims1=range(x.ndim)
        axes=list(where(self.in_mask)[0])
        remain_x=list(where(~self.in_mask)[0])
        remain_tensor=range(x.ndim,x.ndim+tensor.ndim-len(axes))
        return np.einsum(x,dims1,self.tensor,axes+remain_x,remain_x+remain_tensor)

    def backward(self,x,y):
        return dvar,dx

class L_Conv(Layer):
    '''Convolusional Layer.'''
    def __init__(self,fltr,strides,axes):
        self.fltr=np.asarray(fltr)
        self.strides=strides
        self.axes=axes

    def forward(self,x):
        return conv(x,self.fltr,self.strides,axes=self.axes)

    def backward(self,x,y):
        return dvar,dx

    def get_variables(self):
        return self.fltr.ravel()

    def set_variables(self,variables):
        return self.fltr[...]=variables
