'''
Linear Layer.
'''

import numpy as np
import scipy.sparse as sps
import pdb

from core import Layer,EMPTY_ARRAY
from conv import conv

__all__=['L_Tensor','L_Conv']

class L_Tensor(Layer):
    '''
    Tensor Layer.
    
    Attributes:
        :W: ndarray, the weight tensor.
        :einsum_tokens: list, tokens for [x,W,y] in einsum.

    Note:
        dx=W*dy, need conjugate during BP?
    '''
    def __init__(self,W,einsum_tokens):
        self.W=np.asarray(W)
        if len(einsum_tokens)!=3:
            raise ValueError('einsum_tokens should be a len-3 list!')
        if len(einsum_tokens[1])!=self.W.ndim:
            raise ValueError('einsum_tokens dimension error!')
        self.einsum_tokens=einsum_tokens

    def forward(self,x):
        return np.einsum('%s,%s->%s'%tuple(self.einsum_tokens),x,self.W)

    def backward(self,x,y,dy):
        einsum_tokens=self.einsum_tokens
        dx=np.einsum('%s,%s->%s'%tuple(einsum_tokens[::-1]),dy,self.W)
        dW=np.einsum('%s,%s->%s'%(einsum_tokens[0],einsum_tokens[2],einsum_tokens[1]),x,dy)
        return dW,dx

    def get_variables(self):
        return self.W.ravel()

    def set_variables(self,variables):
        if isinstance(variables,np.ndarray):
            variables=variables.reshape(self.W.shape)
        self.W[...]=variables
