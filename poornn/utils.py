from __future__ import division
import numpy as np
import pdb

from .lib import futils

__all__=['take_slice', 'scan2csc', 'typed_random', 'typed_randn', 'typed_uniform', 'tuple_prod',
        'masked_concatenate', 'dtype2token', 'dtype_c2r', 'dtype_r2c', 'complex_backward', 'fsign']

def take_slice(arr,sls,axis):
    '''
    take slices along specific axis.

    Args:
        arr (ndarray): target array.
        sls (slice): the target sector.
        axis (int): the target axis.

    Returns:
        ndarray: result array.
    '''
    return arr[(slice(None),)*axis+(sls,)]

def scan2csc(kernel_shape, img_in_shape, strides, boundary):
    '''
    Scan target shape with filter, and transform it into csc_matrix.

    Args:
        kernel_shape (tuple): shape of kernel.
        img_in_shape (tuple): shape of image dimension.
        strides (tuple): strides for image dimensions.
        boundary ('P'|'O'): boundary condition.

    Returns:
        (1darray, 1darray, tuple): indptr for csc maitrx, indices of csc matrix, output image shape.
    '''
    if len(img_in_shape)!=len(strides) or len(kernel_shape)!=len(strides):
        raise ValueError("Dimension Error! (%d, %d, %d)"%(len(strides),len(img_in_shape),len(kernel_shape)))

    dim_kernel = tuple_prod(kernel_shape)
    # get output image shape
    dimension = len(strides)
    img_out_shape=[]
    for i in range(dimension):
        dim_scan=img_in_shape[i]
        if boundary=='P':
            pass
        elif boundary=='O':
            dim_scan-=kernel_shape[i]-strides[i]
        else:
            raise ValueError("Type of boundary Error!")
        if dim_scan%strides[i]!=0: raise ValueError("Stride and Shape not match!")
        num_sweep_i=dim_scan//strides[i]
        img_out_shape.append(num_sweep_i)
    img_out_shape = tuple(img_out_shape)
    dim_out = tuple_prod(img_out_shape)

    # create a sparse csc_matrix(dim_in, dim_out), used in fortran and start from 1!.
    csc_indptr=np.arange(1,dim_kernel*dim_out+2, dim_kernel, dtype='int32')
    csc_indices=[]   #pointer to rows in x
    for ind_out in range(dim_out):
        ijk_out = np.unravel_index(ind_out, img_out_shape, order='F')
        ijk_in0 = np.asarray(ijk_out)*strides
        for ind_offset in range(tuple_prod(kernel_shape)):
            ijk_in = ijk_in0 + np.unravel_index(ind_offset, kernel_shape, order='F')
            ind_in = np.ravel_multi_index(ijk_in, img_in_shape, mode='wrap' if boundary=='P' else 'raise', order='F')
            csc_indices.append(ind_in+1)

    csc_indices=np.int32(csc_indices)
    return csc_indptr, csc_indices, img_out_shape

def spscan2csc(cscmat, strides):
    '''
    Scan target shape with csc matrix, and transform it into a larger csc matrix,
    boundary condition must be periodic.

    Args:
        cscmat (scipy.sparse.csc_maitrx): the kernel matrix.
        strides (tuple): strides for image dimensions.

    Returns:
        (1darray, 1darray, tuple): indptr for csc maitrx, indices of csc matrix, output image shape.
    '''
    if len(img_in_shape)!=len(strides):
        raise ValueError("Dimension Error! (%d, %d)"%(len(strides),len(img_in_shape)))

    # get output image shape
    dimension = len(strides)
    img_out_shape=[]
    for i in range(dimension):
        dim_scan=img_in_shape[i]
        if dim_scan%strides[i]!=0: raise ValueError("Stride and Shape not match!")
        num_sweep_i=dim_scan//strides[i]
        img_out_shape.append(num_sweep_i)
    img_out_shape = tuple(img_out_shape)
    dim_out = tuple_prod(img_out_shape)

    # create a sparse csc_matrix(dim_in, dim_out), used in fortran and start from 1!.
    csc_indptr=np.arange(1,cscmat.nnz*dim_out+2, cscmat.nnz, dtype='int32')
    csc_indices=[]   #pointer to rows in x
    for ind_out in range(dim_out):
        ijk_out = np.unravel_index(ind_out, img_out_shape, order='F')
        ijk_in0 = np.asarray(ijk_out)*strides
        for ind_offset in range(tuple_prod(kernel_shape)):
            ijk_in = ijk_in0 + np.unravel_index(ind_offset, kernel_shape, order='F')
            ind_in = np.ravel_multi_index(ijk_in, img_in_shape, mode='wrap' if boundary=='P' else 'raise', order='F')
            csc_indices.append(ind_in+1)

    csc_indices=np.int32(csc_indices)
    return csc_indptr, csc_indices, img_out_shape

def typed_random(dtype, shape):
    '''
    generate a random numbers with specific data type.

    Args:
        dtype (str): data type.
        shape (tuple): shape of desired array.

    Returns:
        ndarray: random array in 'F' order.
    '''
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
    generate a normal distributed random numbers with specific data type.

    Args:
        dtype (str): data type.
        shape (tuple): shape of desired array.

    Returns:
        ndarray: random array in 'F' order.
    '''
    #fix shape with dummy index.
    shp=[si if si>=0 else np.random.randint(1,21) for si in shape]

    if dtype=='complex128':
        return np.transpose(np.random.randn(*shape[::-1])+1j*np.random.randn(*shape[::-1]))
    elif dtype=='complex64':
        return np.complex64(typed_randn('complex128', shape))
    else:
        return np.transpose(np.random.randn(*shape[::-1])).astype(np.dtype(dtype))

def typed_uniform(dtype, shape, low=-1., high=1.):
    '''
    generate a uniformly distributed random numbers with specific data type.

    Args:
        dtype (str): data type.
        shape (tuple): shape of desired array.

    Returns:
        ndarray: random array in 'F' order.
    '''
    #fix shape with dummy index.
    shp=[si if si>=0 else np.random.randint(1,21) for si in shape]

    if dtype=='complex128':
        return np.transpose(np.random.uniform(low,high,shape[::-1])+1j*np.random.uniform(low,high,shape[::-1]))
    elif dtype=='complex64':
        return np.complex64(typed_uniform('complex128', shape,low,high))
    else:
        return np.transpose(np.random.uniform(low,high,shape[::-1])).astype(np.dtype(dtype))

def tuple_prod(tp):
    '''
    product over a tuple of numbers.

    Args:
        tp (tuple): the target tuple to product over.

    Returns:
        number: product of tuple.
    '''
    res = 1
    for item in tp:
        res*=item
    return res

def masked_concatenate(vl, mask):
    '''
    concatenate multiple arrays with mask True.

    Args:
        vl (list<ndarray>): arrays.
        mask (list<bool>): masks for arrays.

    Returns:
        ndarray: result array.
    '''
    vl_ = [item for item,maski in zip(vl, mask) if maski]
    dvar=np.concatenate(vl_) if len(vl_)!=0 else np.zeros([0], dtype=vl[0].dtype)
    return dvar

def _connect(g, start, end, arr_shape, dtype, pos='mid'):
    '''utility for Connecting graphviz nodes'''
    if start is None or end is None: return
    kwargs = {}
    def get_node(nodes):
        if pos=='first':
            return nodes[0]
        elif pos=='last':
            return nodes[-1]
        elif pos=='mid':
            return nodes[len(nodes)//2]
        else:
            raise ValueError()

    if not hasattr(start, 'attr'):
        kwargs['ltail'] = start.name
        nodes = start.nodes()
        start = get_node(nodes)
    if not hasattr(end, 'attr'):
        kwargs['lhead'] = end.name
        nodes = end.nodes()
        end = get_node(nodes)
    g.add_edge(start, end, label='<<font point-size="10px">%s</font><br align="center"/>\
<font point-size="10px">%s</font><br align="center"/>>'%(arr_shape, dtype), **kwargs)

def dtype2token(dtype):
    '''
    Parse data type to token.
    
    Args:
        dtype ('float32'|'float64'|'complex64'|'complex128'): data type.

    Returns:
        str: 's'|'d'|'c'|'z'
    '''
    if dtype=='complex128':
        dtype_token = 'z'
    elif dtype=='complex64':
        dtype_token = 'c'
    elif dtype=='float64':
        dtype_token = 'd'
    elif dtype=='float32':
        dtype_token = 's'
    else:
        print("Warning: dtype unrecognized - get %s!"%dtype)
        return 'x'
    return dtype_token

def dtype_c2r(complex_dtype):
    '''
    Get corresponding real data type from complex data type.
    
    Args:
        dtype ('complex64'|'complex128'): data type.

    Returns:
        str: ('float32'|'float64')
    '''
    if complex_dtype=='complex128':
        return 'float64'
    elif complex_dtype=='complex64':
        return 'float32'
    else:
        raise ValueError('Complex data type %s not valid'%complex_dtype)

def dtype_r2c(real_dtype):
    '''
    Get corresponding complex data type from real data type
    Args:
        dtype ('float32'|'float64'): data type.

    Returns:
        str: ('complex64'|'complex128')
    '''
    if real_dtype=='float64':
        return 'complex128'
    elif real_dtype=='float32':
        return 'complex64'
    else:
        raise ValueError('Real data type %s not valid'%real_dtype)

def get_tag(layer, tag):
    '''
    Get tag from a layer,

    Args:
        layer (Layer): target layer.
        tag ('runtimes'|'is_inplace'|'tags'): the desired tag.

    Returns:
        object: tag information for layer.
    '''
    if hasattr(layer, 'tags') and tag in layer.tags:
        return layer.tags[tag]
    else:
        from .core import DEFAULT_TAGS
        if tag in DEFAULT_TAGS:
            return DEFAULT_TAGS[tag]
        else:
            raise KeyError('Can not find Tag %s'%tag)

def complex_backward(dz,dzc):
    '''
    Complex propagation rule.

    Args:
        dz (ndarray): :math:`\partial J/\partial z`
        dzc (ndarray): :math:`\partial J/\partial z^*`

    Returns:
        func: backward function that take xy and dy as input.
    '''
    def backward(xy,dy,**kwargs):
        x, y = xy
        if dz is None:
            return (dzc(x,y)*dy).conj()
        elif dzc is None:
            return dz(x,y)*dy
        else:
            return dz(x,y)*dy+(dzc(x,y)*dy).conj()
    return backward

def fsign(x):
    '''
    sign function that work properly for complex numbers :math:`x/|x|`,

    Args:
        x (ndarray): input array.

    Returns:
        ndarray: sign of x.

    Note:
        if x is 0, return 0.
    '''
    order = 'F' if np.isfortran(x) else 'C'
    return eval('futils.fsign_%s'%dtype2token(x.dtype.name))(x.ravel(order=order)).reshape(x.shape,order=order)

if __name__ == '__main__':
    print(typed_uniform('complex128',(2,2)))
    print(typed_uniform('float64',(2,2)))
    typed_randn('float64',(2,2))
