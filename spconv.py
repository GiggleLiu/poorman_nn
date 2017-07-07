'''
Convolution using sparse matrix.
'''

import numpy as np
import pdb,time

from spconvz import lib as spconvz
from spconvd import lib as spconvd

class SPConv(object):
    '''
    Attributes:
        ::
    '''
    def __init__(self, fltr, img_in_shape, bias, dtype='float32', strides=(1,1), boundary = "P"):
        self.dtype = dtype
        fltr=np.asarray(fltr, dtype = dtype, order = 'C')
        num_feature_in, num_feature_out = fltr.shape[:2]
        kernel_shape = fltr.shape[2:]
        dim_kernel = np.prod(kernel_shape)
        dim_in = np.prod(img_in_shape)
        self.img_in_shape = img_in_shape
        self.fltr = fltr
        self.strides = strides
        self.boundary = boundary
        self.bias = bias
        if dtype=='complex128':
            self._lib=spconvz
        elif dtype=='float64':
            self._lib=spconvd
        elif dtype=='float32':
            self._lib=spconvd
        else:
            raise TypeError("wrong type of dtype!")

        if len(img_in_shape)!=len(strides) or len(kernel_shape)!=len(strides):
            raise ValueError("Dimension Error! (%d, %d, %d, %d)"%(len(strides),len(size_out),len(img_in_shape),len(kernel_shape)))
        dimension = len(strides)

        # get output image shape
        img_out_shape=[]
        for i in xrange(dimension):
            dim_scan=img_in_shape[i]
            if boundary=='P':
                pass
            elif boundary=='O':
                dim_scan-=kernel_shape[i]-1
            else:
                raise ValueError("Type of boundary Error!")
            if dim_scan%strides[i]!=0: raise ValueError("Stride and Shape not match!")
            num_sweep_i=dim_scan/strides[i]
            img_out_shape.append(num_sweep_i)
        dim_out = np.prod(img_out_shape)

        # create a sparse csc_matrix(dim_in, dim_out), used in fortran and start from 1!.
        self.csc_indptr=np.arange(1,dim_kernel*dim_out+2, dim_kernel, dtype='int32')
        csc_indices=[]
        if dimension==1:
            for i in xrange(img_out_shape[0]):
                csc_indices.append(np.arange(strides[0]*i+1,strides[0]*i+dim_filter+1,dtype='int32')%dim_in)
        if dimension==2:
            for i in xrange(img_out_shape[0]):
                for j in xrange(img_out_shape[1]):
                    i0, j0 = i*strides[0], j*strides[1]
                    start_ = i*img_in_shape[1]+j
                    indices_ij = np.concatenate(np.int32([[(i_%img_in_shape[0])*img_in_shape[1]+j_%img_in_shape[1]+1 for j_ in xrange(j0,j0+kernel_shape[1])]\
                            for i_ in xrange(i0,i0+kernel_shape[0])]))
                    csc_indices.append(indices_ij)
        else:
            raise ValueError("Dimension too high!")
        self.csc_indices=np.concatenate(csc_indices)
        #self.fltr_data = np.asfortranarray(np.transpose(np.tile(fltr.reshape([num_feature_in,num_feature_out,dim_kernel]), [1,1,np.prod(img_out_shape)]),axes=(2,0,1)).T)
        self.weight_indices = np.asfortranarray(np.tile(np.arange(dim_kernel,dtype='int32'),np.prod(img_out_shape)))+1
        self.fltr_data = np.asfortranarray(np.transpose(fltr.reshape([num_feature_in,num_feature_out,dim_kernel]),axes=(1,0,2)))
        self.img_out_shape = img_out_shape

    @property
    def NDIM(self):
        return len(self.strides)

    def forward(self, x):
        '''
        Parameters:
            :x: ndarray, (dim_in(s), num_batch, num_feature_in), input in 'C' order.
        Return:
            ndarray, (dim_out(s), num_batch, num_feature_out), output in 'C' order.
        '''
        x=x.reshape([np.prod(x.shape[:-2]),x.shape[-2],x.shape[-1]])
        x=x.T
        #x=x.reshape([x.shape[0],x.shape[1],np.prod(x.shape[-2:])])
        #x=x.reshape([x.shape[0],np.prod(x.shape[1:-1]),x.shape[-1]])
        y=self._lib.forward(x,csc_indptr=self.csc_indptr,csc_indices=self.csc_indices,fltr_data=self.fltr_data,
                weight_indices=self.weight_indices,
                bias=self.bias, max_nnz_row=np.prod(self.fltr.shape[-self.NDIM:]))
        #newshape=(y.shape[0],)+tuple(self.img_out_shape[3])+(y.shape[-1],)
        #newshape=y.shape[:2]+tuple(self.img_out_shape[3])
        y=y.T
        y=y.reshape(tuple(self.img_out_shape[3])+y.shape[-2:])
        return y

    def backward(self, x, y, dy, dx, dweight, dbias, mask=(1,)*3):
        '''
        Parameters:
            :x: ndarray, (dim_in(s), num_batch, num_feature_in), input in 'C' order.
            :y: ndarray, (dim_out(s), num_batch, num_feature_out), output in 'C' order.
            :dy: ndarray, (dim_out(s), num_batch, num_feature_out), gradient of output in 'C' order.
            :dx: ndarray, (dim_in(s), num_batch, num_feature_in), gradient of input in 'C' order, used for accumulate gradient.
            :dweight: ndarray, (nnz, dim_in(s), dim_out(s)), gradient of weight variables in 'C' order, used for accumulate gradient.
            :dbias: ndarray, (num_feature_out), gradient of bias variables in 'C' order, used for accumulate gradient.
            :mask: booleans, (do_xgrad, do_wgrad, do_bgrad).
        '''
        x=x.reshape([np.prod(x.shape[:-2]),x.shape[-2],x.shape[-1]]).T
        y=y.reshape([np.prod(y.shape[:-2]),y.shape[-2],y.shape[-1]]).T
        dx=dx.reshape([np.prod(dx.shape[:-2]),dx.shape[-2],dx.shape[-1]]).T
        dy=dy.reshape([np.prod(dy.shape[:-2]),dy.shape[-2],dy.shape[-1]]).T
        res=self._lib.backward(dy,x,dx,dweight,dbias,self.csc_indptr,self.csc_indices,
                weight_indices=self.weight_indices, fltr_data=self.fltr_data, bias=self.bias, 
                do_xgrad=mask[0], do_wgrad=mask[1], do_bgrad=mask[2], max_nnz_row=np.prod(self.fltr.shape[-self.NDIM:]))
        dx=dx.T
        dx=dx.reshape(tuple(self.img_in_shape)+(dx.shape[-2],dx.shape[-1]))

    def get_zero_gradients(self):
        '''Get empty gradients.'''
        return np.zeros_like(self.fltr_data)

