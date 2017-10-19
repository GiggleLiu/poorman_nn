import numpy as np

from .core import Monitor

__all__ = ['Print', 'PlotStat', 'Cache']

class Print(Monitor):
    '''Print data without changing anything.'''
    def __init__(self, input_shape, itype, **kwargs):
        super(Print, self).__init__(input_shape, input_shape, itype)

    @classmethod
    def monitor_forward(self, x):
        print('Forward\n -  x = %s'%x)
    
    @classmethod
    def monitor_backward(self, xy, dy, **kwargs):
        x,y=xy
        print('Backward\n -  x = %s\n -  y = %s\n -  dy = %s'%(x,y,dy))

class PlotStat(Monitor):
    '''
    Print data without changing anything.

    Args:
        ax (<matplotib.axes>):
        mask (list<bool>, len=2, default=[True,False]): masks for forward check and backward check.

    Attributes:
        ax (<matplotib.axes>):
        mask (list<bool>, len=2): masks for forward check and backward check.
    '''
    def __init__(self, input_shape, itype, ax, mask=[True,False], **kwargs):
        super(PlotStat, self).__init__(input_shape, input_shape, itype)
        self.ax = ax
        self.mask = mask

    def monitor_forward(self, x):
        if not self.mask[0]: return
        xx = np.sort(x.ravel())[::-1]
        self.ax.clear()
        self.ax.bar(np.arange(len(xx)),xx)
        self.ax.title(' X: Mean = %.4f, Std. = %.4f'%(x.mean,np.std(x)))
    
    def monitor_backward(self, xy, dy, **kwargs):
        if not self.mask[1]: return
        xx = np.sort(dy.ravel())[::-1]
        self.ax.clear()
        self.ax.bar(np.arange(len(xx)),xx)
        self.ax.title('DY: Mean = %.4f, Std. = %.4f'%(x.mean,np.std(x)))

class Cache(Monitor):
    '''
    Cache data without changing anything.

    Attributes:
        forward_list (list): cached forward data.
        backward_list (list): cached backward data.
    '''
    def __init__(self, input_shape, itype, **kwargs):
        super(Cache, self).__init__(input_shape, input_shape, itype)
        self.forward_list = []
        self.backward_list = []

    def monitor_forward(self, x,**kwargs):
        self.forward_list.append(x)
    
    def monitor_backward(self, xy, dy, **kwargs):
        self.backward_list.append(dy)

    def clear(self):
        '''clear history.'''
        self.forward_list = []
        self.backward_list = []
