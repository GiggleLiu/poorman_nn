'''
Convolution using sparse matrix.
'''

import numpy as np
import pdb,time

from lib.spconvz import lib as spconvz
from lib.spconvd import lib as spconvd
from utils import scan2csc

class SPConv(object):
    '''
    Attributes:
        :fltr: ndarray, (feature_out, feature_in, kernel_x, ...)
        :bias: 1darray, (feature_out), in fortran order.
        :img_in_shape: tuple, input image shape.
        :dtype: str, data type.
        :strides: tuple, displace for convolutions.
        :boudnary: choice('P', 'O').
            * 'P', periodic boundary condiction.
            * 'O', open boundary condition.

    Attributes (Derived):
        :img_out_shape: tuple, output image shape.
        :csc_indptr: 1darray, column pointers for convolution matrix.
        :csc_indices: 1darray, row indicator for input array.
        :weight_indices: 1darray, row indicator for filter array (if not contiguous).
        :_fltr_flatten: 3darray, flattened filter, in fortran order.
    '''
    def __init__(self, fltr, bias, img_in_shape, dtype='float32', strides=(1,1), boundary = "P", w_contiguous = True):
        #set data type
        self.dtype = dtype
        if dtype=='complex128':
            self._lib=spconvz
        elif dtype=='float64':
            self._lib=spconvd
        elif dtype=='float32':
            self._lib=spconvd
        else:
            raise TypeError("wrong type of dtype!")

        self.fltr = np.asarray(fltr, dtype = dtype)
        self.bias = np.asarray(bias, order = 'F', dtype = dtype)
        self._fltr_flatten = np.asarray(np.reshape(fltr,fltr.shape[:2]+(-1,)), dtype = dtype, order = 'F')

        self.img_in_shape = tuple(img_in_shape)
        self.strides = tuple(strides)
        self.boundary = boundary
        self.w_contiguous = w_contiguous

        num_feature_out, num_feature_in = self.fltr.shape[:2]
        kernel_shape = self.fltr.shape[2:]

        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(kernel_shape, img_in_shape, strides, boundary)
        #self.fltr_data = np.asfortranarray(np.transpose(np.tile(fltr.reshape([num_feature_in,num_feature_out,dim_kernel]), [1,1,np.prod(self.img_out_shape)]),axes=(2,0,1)).T)
        if not w_contiguous:
            self.weight_indices = np.asfortranarray(np.tile(np.arange(np.prod(kernel_shape),dtype='int32'),np.prod(self.img_out_shape)))+1  #pointer to filter data

    @property
    def img_nd(self):
        '''Dimension of input image.'''
        return len(self.strides)

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (dim_in(s), num_feature_in, num_batch), input in 'C' order.
        Return:
            ndarray, (dim_out(s), num_feature_out, num_batch), output in 'C' order.
        '''
        if x.ndim == self.img_nd + 1:  #single batch wise
            return self._forward_singlebatch(x)
        x=x.reshape([np.prod(self.img_in_shape),x.shape[-2],x.shape[-1]])
        x=x.T
        if self.w_contiguous:
            y=self._lib.forward_contiguous(x,csc_indptr=self.csc_indptr,csc_indices=self.csc_indices,fltr_data=self._fltr_flatten,
                    bias=self.bias, max_nnz_row=self._fltr_flatten.shape[-1])
        else:
            y=self._lib.forward_general(x,csc_indptr=self.csc_indptr,csc_indices=self.csc_indices,fltr_data=self._fltr_flatten,
                    weight_indices=self.weight_indices,
                    bias=self.bias, max_nnz_row=self._fltr_flatten.shape[-1])
        y=y.T
        y=y.reshape(self.img_out_shape+y.shape[-2:])
        return y

    def backward(self, x, y, dy, dx, dweight, dbias, mask=(1,)*3):
        '''
        Parameters:
            :x: ndarray, (dim_in(s), num_feature_in, num_batch), input in 'C' order.
            :y: ndarray, (dim_out(s), num_feature_out, num_batch), output in 'C' order.
            :dy: ndarray, (dim_out(s), num_feature_out, num_batch), gradient of output in 'C' order.
            :dx: ndarray, (dim_in(s),, num_feature_in num_batch), gradient of input in 'C' order, used for accumulate gradient.
            :dweight: ndarray, (nnz, dim_in(s), dim_out(s)), gradient of weight variables in 'C' order, used for accumulate gradient.
            :dbias: ndarray, (num_feature_out), gradient of bias variables in 'C' order, used for accumulate gradient.
            :mask: booleans, (do_xgrad, do_wgrad, do_bgrad).
        '''
        if x.ndim == self.img_nd + 1:  #single batch wise
            return self._backward_singlebatch(x, y, dy, dx, dweight, dbias, mask)
        dim_in=np.prod(self.img_in_shape)
        dim_out=np.prod(self.img_out_shape)
        x=x.reshape([dim_in,x.shape[-2],x.shape[-1]]).T
        y=y.reshape([dim_out,y.shape[-2],y.shape[-1]]).T
        dx=dx.reshape([dim_in,dx.shape[-2],dx.shape[-1]]).T
        dy=dy.reshape([dim_out,dy.shape[-2],dy.shape[-1]]).T
        if self.w_contiguous:
            res=self._lib.backward_contiguous(dy,x,dx,dweight,dbias,self.csc_indptr,self.csc_indices,
                    fltr_data=self._fltr_flatten, bias=self.bias, 
                    do_xgrad=mask[0], do_wgrad=mask[1], do_bgrad=mask[2], max_nnz_row=self._fltr_flatten.shape[-1])
        else:
            res=self._lib.backward_general(dy,x,dx,dweight,dbias,self.csc_indptr,self.csc_indices,
                    weight_indices=self.weight_indices, fltr_data=self._fltr_flatten, bias=self.bias, 
                    do_xgrad=mask[0], do_wgrad=mask[1], do_bgrad=mask[2], max_nnz_row=self._fltr_flatten.shape[-1])
        dx=dx.T
        dx=dx.reshape(self.img_in_shape+(dx.shape[-2],dx.shape[-1]))

    def get_zero_gradients(self):
        '''Get empty gradients.'''
        return np.zeros_like(self.fltr)

    def _forward_singlebatch(self, x):
        x=x.reshape([np.prod(self.img_in_shape),x.shape[-1]])
        x=x.T
        if self.w_contiguous:
            y=self._lib.forward1_contiguous(x,csc_indptr=self.csc_indptr,csc_indices=self.csc_indices,fltr_data=self._fltr_flatten,
                    bias=self.bias, max_nnz_row=self._fltr_flatten.shape[-1])
        else:
            y=self._lib.forward1_general(x,csc_indptr=self.csc_indptr,csc_indices=self.csc_indices,fltr_data=self._fltr_flatten,
                    weight_indices=self.weight_indices,
                    bias=self.bias, max_nnz_row=self._fltr_flatten.shape[-1])
        y=y.T
        y=y.reshape(self.img_out_shape+y.shape[-1:])
        return y


    def _backward_singlebatch(self, x, y, dy, dx, dweight, dbias, mask=(1,)*3):
        dim_in=np.prod(self.img_in_shape)
        dim_out=np.prod(self.img_out_shape)
        x=x.reshape([dim_in,x.shape[-1]]).T
        y=y.reshape([dim_out,y.shape[-1]]).T
        dx=dx.reshape([dim_in,dx.shape[-1]]).T
        dy=dy.reshape([dim_out,dy.shape[-1]]).T
        if self.w_contiguous:
            res=self._lib.backward1_contiguous(dy,x,dx,dweight,dbias,self.csc_indptr,self.csc_indices,
                    fltr_data=self._fltr_flatten, bias=self.bias, 
                    do_xgrad=mask[0], do_wgrad=mask[1], do_bgrad=mask[2], max_nnz_row=self._fltr_flatten.shape[-1])
        else:
            res=self._lib.backward1_general(dy,x,dx,dweight,dbias,self.csc_indptr,self.csc_indices,
                    weight_indices=self.weight_indices, fltr_data=self._fltr_flatten, bias=self.bias, 
                    do_xgrad=mask[0], do_wgrad=mask[1], do_bgrad=mask[2], max_nnz_row=self._fltr_flatten.shape[-1])
        dx=dx.T
        dx=dx.reshape(self.img_in_shape+dx.shape[-1:])
