'''
Convolusional neural network.
'''

import numpy as np

from utils import take_slice

__all__=['zero_padding','pbc_fill_padding','pbc_padding']

def zero_padding(tensor,extended_dims):
    '''
    Parameters:
        :tensor: ndarray,
        :extended_dims: 2darray, extended dimensions in each direction.

            e.g. [[1,2],   #add 1/2 entry at the start/end of 0-th dimension
                [3,4]]     #add 3/4 entry at the start/end of 1-th dimension

            Note: padding should never be greater than the size of specific dimension!

    Return:
        tensor, dict, dict contains the padding information.
    '''
    tshape=tensor.shape
    padded_tensor=np.zeros(np.array(tshape)+np.sum(extended_dims,axis=1),dtype=tensor.dtype)
    padded_tensor[tuple(slice(start,tshape[i]+start) for i,(start,end) in enumerate(extended_dims))]=tensor
    padding_info={'type':'zero_padding','extended_dims':extended_dims}
    return padded_tensor,padding_info

def pbc_padding(tensor,extended_dims):
    padded_tensor=tensor
    for i,(start,end) in enumerate(extended_dims):
        preslice=(slice(None),)*i
        mats=[]
        if start!=0:
            mats.append(padded_tensor[preslice+(slice(-start,None),)])
        mats.append(padded_tensor)
        if end!=0:
            mats.append(padded_tensor[preslice+(slice(None,end),)])
        padded_tensor=np.concatenate(mats,axis=i)
    return padded_tensor

def pbc_fill_padding(zero_padded_tensor,padding_info):
    '''
    Parameters:
        :zero_padded_tensor: ndarray,
        :padding_info: dict,

        Note: it will change padding_info!
    '''
    padding_info['type']='pbc_padding'
    extended_dims=padding_info['extended_dims']
    _pbc_fill_padding(zero_padded_tensor,extended_dims)


def _pbc_fill_padding(zero_padded_tensor,extended_dims):
    '''
    Parameters:
        :zero_padded_tensor: ndarray,
        :extended_dims: 2darray, see @zero_padding.
    '''
    remaining_dims=len(extended_dims)
    start,end=extended_dims[0]
    preslice=(slice(None),)*(zero_padded_tensor.ndim-remaining_dims)
    if remaining_dims!=1:
        _pbc_fill_padding(zero_padded_tensor[preslice+(slice(start,-end),)],extended_dims[1:])
    zero_padded_tensor[preslice+(slice(None,start),)]=zero_padded_tensor[preslice+(slice(-end-start,-end),)]
    zero_padded_tensor[preslice+(slice(-end,None),)]=zero_padded_tensor[preslice+(slice(start,start+end),)]

def conv(tensor,fltr,strides,axes):
    '''
    Parameters:
        :tensor: ndarray,
        :fltr: ndarray, filter.
    '''
    axes=list(axes)
    #check strides and filter.
    for axis,stride,fltr_shape in zip(axes,strides,fltr.shape):
        if (tensor.shape[axis]-fltr_shape)%stride!=0 or tensor.shape[axis]<fltr_shape:
            raise ValueError()
        
    out_tensor=zeros([dim if not idim in axes else 1+(dim-fltr.shape[axes.index(idim)])/strides[axes.index(idim)]\
            for idim,dim in enumerate(tensor.shape)],dtype=tensor.dtype)
    _conv(tensor,fltr,output=out_tensor,strides=strides,axes=axes)
    return out_tensor

def _conv(tensor,fltr,output,strides,axes):
    axis=axes[0]
    stride=strides[0]
    if len(axes)==1:
        for start in xrange(0,tensor.shape[axis]-fltr.shape[axis]+1,stride):
            output[slicer+(i,)]=tensor[slicer+(slice(start,start+fltr.shape[axis]),)]*fltr
    if axis==tensor.ndim:
        _conv(tensor,output,strides,axis=i+1)
