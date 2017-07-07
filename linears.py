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

class L_Conv(Layer):
    '''
    Convolusional Layer, the einsum version.
    
    Parameters:
        :filters: ndarray, filters for features, 
            Tensor(conv_dim_1,conv_dim_2,...,out_dim_1,out_dim_2,...,feature_dim),
            out_dims should be dummy 1.
        :strides: tuple,
        :num_strides: tuple, the ouput_shape in convolusion directions, input_shape = num_strides*strides.
        :boundary_condition: str, 'preodic'.
    '''
    def __init__(self,filters,strides,num_strides,einsum_tokens,boundary_condition='periodic'):
        if not boundary_condition in ['zero','periodic']:
            raise ValueError('undefined boundary condition!')
        self.filters=np.asarray(filters)
        self.strides=strides
        self.num_strides=num_strides
        self.einsum_tokens=einsum_tokens
        self.boundary_condition=boundary_condition

        #initialize start points and reference table for indices
        self._start_points=self._get_start_points()

    @property
    def num_features(self): return len(self.filters)

    def forward(self,x):
        #extract index information from eincodes
        axes_x_conv,axes_x_remain,axes_y_conv,axes_y_remain=self._decode_einsum()

        #pad x.
        padding=[(0,0)]*x.ndim
        for iaxis,(num_stride,stride,axis) in enumerate(zip(self.num_strides,self.strides,axes_x_conv)):
            padding[axis]=(0,(num_stride*stride+self.filters.shape[iaxis]-1)-x.shape[axis])
        x=np.pad(x,padding,mode='wrap')

        #multiplication
        y_shape=[x.shape[axis] for axis in axes_x_remain]+list(self.num_strides)+[self.filters.shape[-1]]
        y=np.empty(y_shape,dtype=np.find_common_type([x.dtype,self.filters.dtype],[]))

        for istart,start in enumerate(self._start_points):
            #take the target space of x, y
            sls_x=[slice(None)]*x.ndim
            for start_i,axis_i,filters_size_i in zip(start,axes_x_conv,self.filters.shape)[::-1]:
                sls_x[axis_i]=slice(start_i,start_i+filters_size_i)
            sls_y=[slice(None)]*y.ndim
            ind_y_conv=np.unravel_index(istart,self.num_strides)
            for iaxis,ind_i in zip(axes_y_conv,ind_y_conv):
                sls_y[iaxis]=slice(ind_i,ind_i+1)

            #change target space of y
            y[sls_y]=np.einsum('%s,%s->%s'%tuple(self.einsum_tokens),x[sls_x],self.filters)
        return y

    def backward(self,x,y,dy):
        einsum_tokens=self.einsum_tokens
        #extract index information from eincodes
        axes_x_conv,axes_x_remain,axes_y_conv,axes_y_remain=self._decode_einsum()

        #pad x.
        padding=[(0,0)]*x.ndim
        for iaxis,(num_stride,stride,axis) in enumerate(zip(self.num_strides,self.strides,axes_x_conv)):
            padding[axis]=(0,(num_stride*stride+self.filters.shape[iaxis]-1)-x.shape[axis])
        x=np.pad(x,padding,mode='wrap')

        #multiplication
        dx=np.zeros_like(x)
        dW=np.zeros_like(self.filters)

        for istart,start in enumerate(self._start_points):
            #take the target space of x, y
            sls_x=[slice(None)]*x.ndim
            for start_i,axis_i,filter_size_i in zip(start,axes_x_conv,self.filters.shape)[::-1]:
                sls_x[axis_i]=slice(start_i,start_i+filter_size_i)
            sls_y=[slice(None)]*y.ndim
            ind_y_conv=np.unravel_index(istart,self.num_strides)
            for iaxis,ind_i in zip(axes_y_conv,ind_y_conv):
                sls_y[iaxis]=slice(ind_i,ind_i+1)

            dx[sls_x]+=np.einsum('%s,%s->%s'%tuple(einsum_tokens[::-1]),dy[sls_y],self.filters)
            dW+=np.einsum('%s,%s->%s'%(einsum_tokens[0],einsum_tokens[2],einsum_tokens[1]),x[sls_x],dy[sls_y])

        #zip dx
        for axis in axes_x_conv:
            x_size=dx.shape[axis]
            pad_size=padding[axis][1]
            if pad_size!=0:
                dx[(slice(None),)*axis+(slice(0,pad_size),)]+=dx[(slice(None),)*axis+(slice(x_size-pad_size,x_size),)]
                dx=dx[(slice(None),)*axis+(slice(0,x_size-pad_size),)]
        return dW,dx

    def get_variables(self):
        return self.filters.ravel()

    def set_variables(self,variables):
        if isinstance(variables,np.ndarray):
            variables=variables.reshape(self.filters.shape)
        self.filters[...]=variables

    def _get_start_points(self):
        start_points=np.meshgrid(*[np.arange(num_stride)*stride for num_stride,stride in zip(self.num_strides,self.strides)],indexing='ij')
        return np.concatenate([s[...,np.newaxis] for s in start_points],axis=-1)

    def _decode_einsum(self):
        token_x,token_w,token_y=self.einsum_tokens
        nx,ny=len(token_x),len(token_y)
        axes_x_conv,axes_x_diff,axes_y_conv,axes_y_diff=[],[],[],[]
        for ix,tx in enumerate(token_x):
            if tx in token_w:
                axes_x_conv.append(ix)
            else:
                axes_x_diff.append(ix)
        for iy,ty in enumerate(token_y):
            if ty in token_w:
                axes_y_conv.append(iy)
            else:
                axes_y_diff.append(iy)
        return axes_x_conv,axes_x_diff,axes_y_conv,axes_y_diff

class L_Conv_S(Layer):
    '''
    Convolusional Layer, the sparse matrix version.
    
    Parameters:
        :filters: ndarray, filters for features, Tensor(conv_dim_1,conv_dim_2,...,feature_dim).
        :strides: tuple,
        :num_strides: tuple, the ouput_shape in convolusion directions, input_shape = num_strides*strides.
        :boundary_condition: str, 'preodic'.
    '''
    def __init__(self,filters,strides,num_strides,einsum_tokens,boundary_condition='preodic'):
        if not boundary_condition in ['zero','periodic']:
            raise ValueError('undefined boundary condition!')
        self.filters=np.asarray(filters)
        self.strides=strides
        self.einsum_tokens=einsum_tokens
        self.boundary_condition=boundary_condition

        #initialize start points and reference table for indices
        self._start_points,self._ref_table=self._gen_ref_table()

        #initialize sparse matrices.
        self._update_sps_matrices()

    @property
    def num_features(self): return len(self.filters)

    def forward(self,x):
        #reshape x.
        axes=[itk for itk,tk in enumerate(einsum_tokens[0]) if itk in einsum_tokens[1]]
        remain_dims=[xshape for ix,xshape in enumerate(x.shape) if ix not in axes]
        x=np.transpose(x,remain_dims+axes)
        x=x.reshape([-1,np.prod(x.shape[-len(axes):])])

        #multiplication
        return self._csc_filter.__rmul__(x).reshape(remain_dims+self._yshapes)

    def backward(self,x,y,dy):
        #reshape y.
        axes=self.axes
        shape=self.shape
        remain_dims=[xshape for ix,xshape in enumerate(x.shape) if ix not in axes]
        dy=dy.reshape([-1,shape[1]])

        #multiplication
        dx=self._csr_filter.T.__rmul__(dy).reshape(remain_dims+[x.shape[iaxis] for iaxis in axes])
        dvar=x.T*dy
        return dvar,dx

    def get_variables(self):
        return self.filters.ravel()

    def set_variables(self,variables):
        if isinstance(variables,np.ndarray):
            variables=variables.reshape(self.filters.shape)
        self.filters[...]=variables
        self._update_sps_matrices()

    def _update_sps_matrices(self):
        shape=self.shape
        for ifltr,fltr in enumerate(self.filters):
            for i in xrange(ifltr*shape[1]):
                xs.append()
        ys=np.repeat(np.arange(size_output),size_filter)
        self._csc_filter=sps.coo_matrix((data,(xs,ys)),shape=self.shape).tocsc()
        self._csr_filter=self._csc_filter.tocsr()

    def _take_block(self,tensor,blockid,axes):
        Ns=self.filters.shape[1:]
        start=self._start_points[blockid]
        ind=[slice(None)]*ndim(tensor)
        for starti,Ni,axis in zip(start,Ns,axes):
            ind[axis]=slice(starti,starti+Ni)
        return tensor[ind]

    def _gen_ref_table(self):
        '''Generate a big enough reference table for indexing.'''
        Ns=self.filters.shape[:1]
        ref_table=np.arange(np.prod(self.shape)).reshape(self.shape)
        for iaxis,Ni in enumerate(Ns):
            ref_table=np.concatenate([ref_table,ref_table[:Ni-1]],axis=iaxis)
        return start_points,ref_table
