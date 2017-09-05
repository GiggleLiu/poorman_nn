import numpy as np
import pdb

from .core import Layer

__all__=['PReLU']

class PReLU(Layer):
    '''
    Parametric ReLU.
    '''
    def __init__(self, leak = 0, itype='float32'):
        self.leak = leak
        self.itype=itype
        if leak>1 or leak<0:
            raise ValueError('leak parameter should be 0-1!')

    def __call__(self,x):
        return self.forward(x)

    def forward(self, x):
        if self.leak==0:
            return maximum(x,0)
        else:
            return maximum(x,self.leak*x)

    def backward(self, x, y, dy, **kwargs):
        dx=dy.copy(order='F')
        xmask=x<0
        if self.leak==0:
            dx[xmask]=0
        else:
            dx[xmask]=leak*dy
        da = np.sum(dy[xmask]*x[xmask].conj())
        return np.array([da], dtype=self.itype), dx

    def get_variables(self):
        return np.array([self.leak], dtype=self.itype)

    def set_variables(self, a):
        self.leak=a[0]

    @property
    def num_variables(self):
        return 1
