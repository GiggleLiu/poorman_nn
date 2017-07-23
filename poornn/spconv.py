'''
Convolution using sparse matrix.
'''

import numpy as np
import pdb,time

from lib.spconv import lib as fspconv
from utils import scan2csc
from core import Layer, check_shape

class SPConv(Layer):
    '''
    Attributes:
        :fltr: ndarray, (feature_out, feature_in, kernel_x, ...), in fortran order.
        :bias: 1darray, (feature_out), in fortran order.
        :dtype: str, data type.
        :strides: tuple, displace for convolutions.
        :boudnary: choice('P', 'O').
            * 'P', periodic boundary condiction.
            * 'O', open boundary condition.

    Attributes (Derived):
        :csc_indptr: 1darray, column pointers for convolution matrix.
        :csc_indices: 1darray, row indicator for input array.
        :weight_indices: 1darray, row indicator for filter array (if not contiguous).
    '''
    def __init__(self, input_shape, fltr, bias, output_shape=None, dtype='float32', strides=None, boundary = "P", w_contiguous = True):
        #set data type
        self.dtype = dtype

        self.fltr = np.asarray(fltr, dtype = dtype, order='F')
        self.bias = np.asarray(bias, order = 'F', dtype = dtype)

        img_nd = self.fltr.ndim-2
        if strides is None:
            strides=(1,)*img_nd
        else:
            self.strides = tuple(strides)
        self.boundary = boundary
        self.w_contiguous = w_contiguous

        kernel_shape = self.fltr.shape[2:]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(kernel_shape, input_shape[-img_nd:], strides, boundary)
        if output_shape is None:
            output_shape = input_shape[:-img_nd-1]+(self.num_feature_out,)+self.img_out_shape
        super(SPConv, self).__init__(input_shape, output_shape)

        #use the correct fortran subroutine.
        if dtype=='complex128':
            dtype_token = 'z'
        elif dtype=='complex64':
            dtype_token = 'c'
        elif dtype=='float64':
            dtype_token = 'd'
        elif dtype=='float32':
            dtype_token = 's'
        else:
            raise TypeError("dtype error!")

        if not w_contiguous:
            self.weight_indices = np.asfortranarray(np.tile(np.arange(np.prod(kernel_shape),dtype='int32'),np.prod(img_out_shape)))+1  #pointer to filter data
            func_f=eval('fspconv.forward_general%s'%dtype_token)
            func_b=eval('fspconv.backward_general%s'%dtype_token)
            func1_f=eval('fspconv.forward1_general%s'%dtype_token)
            func1_b=eval('fspconv.backward1_general%s'%dtype_token)
            self._fforward=lambda *args,**kwargs:func_f(*args,weight_indices=self.weight_indices,**kwargs)
            self._fbackward=lambda *args,**kwargs:func_b(*args,weight_indices=self.weight_indices,**kwargs)
            self._fforward1=lambda *args,**kwargs:func1_f(*args,weight_indices=self.weight_indices,**kwargs)
            self._fbackward1=lambda *args,**kwargs:func1_b(*args,weight_indices=self.weight_indices,**kwargs)
        else:
            self._fforward=eval('fspconv.forward_contiguous%s'%dtype_token)
            self._fforward1=eval('fspconv.forward1_contiguous%s'%dtype_token)
            self._fbackward=eval('fspconv.backward_contiguous%s'%dtype_token)
            self._fbackward1=eval('fspconv.backward1_contiguous%s'%dtype_token)

    def __str__(self):
        return self.__repr__()+'\n  dtype = %s\n  filter => %s\n  bias => %s'%(self.dtype,self.fltr.shape,self.bias.shape)

    @property
    def img_nd(self):
        '''Dimension of input image.'''
        return len(self.strides)

    @property
    def num_feature_in(self):
        '''Dimension of input feature.'''
        return self.fltr.shape[1]

    @property
    def num_feature_out(self):
        '''Dimension of input feature.'''
        return self.fltr.shape[0]

    @check_shape((1,))
    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), input in 'F' order.
        Return:
            ndarray, (num_batch, nfo, img_out_dims), output in 'F' order.
        '''
        x_nd, img_nd = x.ndim, self.img_nd
        self._check_input(x)

        #flatten inputs/outputs
        x=x.reshape(x.shape[:x_nd-img_nd]+(-1,),order='F')
        _fltr_flatten = self.fltr.reshape(self.fltr.shape[:2]+(-1,), order='F')

        if x_nd == img_nd + 1:  #single batch wise
            y=self._fforward1(x,csc_indptr=self.csc_indptr,csc_indices=self.csc_indices,fltr_data=_fltr_flatten,
                    bias=self.bias, max_nnz_row=_fltr_flatten.shape[-1])
        else:
            y=self._fforward(x,csc_indptr=self.csc_indptr,csc_indices=self.csc_indices,fltr_data=_fltr_flatten,
                    bias=self.bias, max_nnz_row=_fltr_flatten.shape[-1])
        y=y.reshape(self.output_shape, order='F')
        return y

    @check_shape((1,-3))
    def backward(self, x, y, dy, mask=(1,)*2):
        '''
        Parameters:
            :x: ndarray, (num_batch, nfi, img_in_dims), input in 'F' order.
            :y: ndarray, (num_batch, nfo, img_out_dims), output in 'F' order.
            :dy: ndarray, (num_batch, nfo, img_out_dims), gradient of output in 'F' order.
            :mask: booleans, (do_xgrad, do_wgrad, do_bgrad).

        Return:
            (dweight, dbias), dx
        '''
        x_nd, img_nd = x.ndim, self.img_nd
        xpre=x.shape[:x_nd-img_nd]
        ypre=xpre[:-1]+(self.num_feature_out,)

        #flatten inputs/outputs
        x=x.reshape(xpre+(-1,), order='F')
        dy=dy.reshape(ypre+(-1,), order='F')
        _fltr_flatten = self.fltr.reshape(self.fltr.shape[:2]+(-1,), order='F')

        if x_nd == img_nd + 1:  #single batch wise
            dx, dweight, dbias = self._fbackward1(dy,x,self.csc_indptr,self.csc_indices,
                    fltr_data=_fltr_flatten,
                    do_xgrad=mask[1], do_wgrad=mask[0], do_bgrad=mask[0], max_nnz_row=_fltr_flatten.shape[-1])
        else:
            dx, dweight, dbias = self._fbackward(dy,x,self.csc_indptr,self.csc_indices,
                    fltr_data=_fltr_flatten,
                    do_xgrad=mask[1], do_wgrad=mask[0], do_bgrad=mask[0], max_nnz_row=_fltr_flatten.shape[-1])
        dx=dx.reshape(self.input_shape, order='F')
        dweight = dweight.reshape(self.fltr.shape, order='F')
        return (dweight, dbias), dx

    def get_variables(self):
        return (self.fltr,self.bias)

    def set_variables(self, variables, mode='set'):
        if mode=='set':
            #self.fltr[...]=variables[0]
            #self.bias[...]=variables[1]
            self.fltr, self.bias = variables
        elif mode=='add':
            self.fltr+=variables[0]
            self.bias+=variables[1]

    @property
    def num_variables(self):
        return 2
