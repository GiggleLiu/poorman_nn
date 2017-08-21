from operator import mul
import numpy as np
import pdb

__all__=['take_slice', 'scan2csc', 'typed_random', 'typed_randn', 'tuple_prod', 'masked_concatenate']

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
        raise ValueError("Dimension Error! (%d, %d, %d)"%(len(strides),len(img_in_shape),len(kernel_shape)))

    dim_kernel = tuple_prod(kernel_shape)
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
    dim_out = tuple_prod(img_out_shape)

    # create a sparse csc_matrix(dim_in, dim_out), used in fortran and start from 1!.
    csc_indptr=np.arange(1,dim_kernel*dim_out+2, dim_kernel, dtype='int32')
    csc_indices=[]   #pointer to rows in x
    for ind_out in xrange(dim_out):
        ijk_out = np.unravel_index(ind_out, img_out_shape, order='F')
        ijk_in0 = np.asarray(ijk_out)*strides
        for ind_offset in xrange(tuple_prod(kernel_shape)):
            ijk_in = ijk_in0 + np.unravel_index(ind_offset, kernel_shape, order='F')
            ind_in = np.ravel_multi_index(ijk_in, img_in_shape, mode='wrap' if boundary=='P' else 'raise', order='F')
            csc_indices.append(ind_in+1)

    csc_indices=np.int32(csc_indices)
    return csc_indptr, csc_indices, img_out_shape

def spscan2csc(cscmat, strides):
    if len(img_in_shape)!=len(strides):
        raise ValueError("Dimension Error! (%d, %d)"%(len(strides),len(img_in_shape)))

    # get output image shape
    dimension = len(strides)
    img_out_shape=[]
    for i in xrange(dimension):
        dim_scan=img_in_shape[i]
        if dim_scan%strides[i]!=0: raise ValueError("Stride and Shape not match!")
        num_sweep_i=dim_scan/strides[i]
        img_out_shape.append(num_sweep_i)
    img_out_shape = tuple(img_out_shape)
    dim_out = tuple_prod(img_out_shape)

    # create a sparse csc_matrix(dim_in, dim_out), used in fortran and start from 1!.
    csc_indptr=np.arange(1,cscmat.nnz*dim_out+2, cscmat.nnz, dtype='int32')
    csc_indices=[]   #pointer to rows in x
    for ind_out in xrange(dim_out):
        ijk_out = np.unravel_index(ind_out, img_out_shape, order='F')
        ijk_in0 = np.asarray(ijk_out)*strides
        for ind_offset in xrange(tuple_prod(kernel_shape)):
            ijk_in = ijk_in0 + np.unravel_index(ind_offset, kernel_shape, order='F')
            ind_in = np.ravel_multi_index(ijk_in, img_in_shape, mode='wrap' if boundary=='P' else 'raise', order='F')
            csc_indices.append(ind_in+1)

    csc_indices=np.int32(csc_indices)
    return csc_indptr, csc_indices, img_out_shape


def pack_variables(variables):
    '''Pack tuple of variables to vector.'''
    shapes = [v.shape for v in variables]
    return np.concatenate([v.ravel(order='F') for v in variables]), shapes

def unpack_variables(vec, shapes):
    '''Unpack vector to tuple of variables.'''
    start = 0
    variables = []
    for shape in shapes:
        end = start+tuple_prod(shape)
        variables.append(vec[start:end].reshape(shape, order='F'))
        start=end
    return variables

def typed_random(dtype, shape):
    #fix shape with dummy index.
    shp=[si if si>=0 else np.random.randint(1,21) for si in shape]

    if dtype=='complex128':
        return np.transpose(np.random.random(shape[::-1])+1j*np.random.random(shape[::-1]))
    elif dtype=='complex64':
        return np.complex64(typed_random('complex128', shape))
    else:
        return np.transpose(np.random.random(shape[::-1])).astype(np.dtype(dtype))

def typed_randn(dtype, shape):
    '''
    Typed random normal distributions, in fortran order.
    '''
    #fix shape with dummy index.
    shp=[si if si>=0 else np.random.randint(1,21) for si in shape]

    if dtype=='complex128':
        return np.transpose(np.random.randn(*shape[::-1])+1j*np.random.randn(*shape[::-1]))
    elif dtype=='complex64':
        return np.complex64(typed_randn('complex128', shape))
    else:
        return np.transpose(np.random.randn(*shape[::-1])).astype(np.dtype(dtype))

tuple_prod = lambda tp: reduce(mul,tp,1)

def masked_concatenate(vl, mask):
    '''concatenate multiple arrays only with those masked.'''
    vl_ = [item for item,maski in zip(vl, mask) if maski]
    dvar=np.concatenate(vl_) if len(vl_)!=0 else np.zeros([0], dtype=vl[0].dtype)
    return dvar

if __name__ == '__main__':
    typed_randn('float64',(2,2))
