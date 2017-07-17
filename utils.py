import numpy as np
import pdb

__all__=['take_slice', 'scan2csc']

def take_slice(arr,sls,axis):
    '''take using slices.'''
    return arr[(slice(None),)*axis+(sls,)]

def scan2csc(kernel_shape, img_in_shape, strides, boundary):
    '''
    Scan target shape with filter, and transform it into csc_matrix.

    Parameters:
        :kernel_shape: tuple,
        :img_in_shape: tuple,
        :strides: tuple,
        :boundary: str,
    '''
    if len(img_in_shape)!=len(strides) or len(kernel_shape)!=len(strides):
        raise ValueError("Dimension Error! (%d, %d, %d, %d)"%(len(strides),len(size_out),len(img_in_shape),len(kernel_shape)))

    dim_kernel = np.prod(kernel_shape)
    # get output image shape
    dimension = len(strides)
    img_out_shape=[]
    for i in xrange(dimension):
        dim_scan=img_in_shape[i]
        if boundary=='P':
            pass
        elif boundary=='O':
            dim_scan-=kernel_shape[i]-strides[i]
        else:
            raise ValueError("Type of boundary Error!")
        if dim_scan%strides[i]!=0: raise ValueError("Stride and Shape not match!")
        num_sweep_i=dim_scan/strides[i]
        img_out_shape.append(num_sweep_i)
    img_out_shape = tuple(img_out_shape)
    dim_out = np.prod(img_out_shape)

    # create a sparse csc_matrix(dim_in, dim_out), used in fortran and start from 1!.
    csc_indptr=np.arange(1,dim_kernel*dim_out+2, dim_kernel, dtype='int32')
    csc_indices=[]   #pointer to rows in x
    for ind_out in xrange(dim_out):
        ijk_out = np.unravel_index(ind_out, img_out_shape, order='F')
        ijk_in0 = np.asarray(ijk_out)*strides
        for ind_offset in xrange(np.prod(kernel_shape)):
            ijk_in = ijk_in0 + np.unravel_index(ind_offset, kernel_shape, order='F')
            ind_in = np.ravel_multi_index(ijk_in, img_in_shape, mode='wrap' if boundary=='P' else 'raise', order='F')
            csc_indices.append(ind_in+1)

    csc_indices=np.int32(csc_indices)
    return csc_indptr, csc_indices, img_out_shape
