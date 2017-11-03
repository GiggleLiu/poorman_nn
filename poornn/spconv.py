'''
Convolution using sparse matrix.
'''
from __future__ import division

import numpy as np
import pdb
import time
from scipy import sparse as sps

from .lib.spconv import lib as fspconv
from .lib.spsp import lib as fspsp
from .utils import scan2csc, tuple_prod, spscan2csc,\
    masked_concatenate, dtype2token, typed_randn
from .linears import LinearBase

__all__ = ['SPConv']


class SPConv(LinearBase):
    '''
    Convolution layer.

    Args:
        weight (ndarray): dimensions are aranged as\
                (feature_out, feature_in, kernel_x, ...), in 'F' order.
        bias (1darray): length of num_feature_out.
        strides (tuple, default=(1,1,...)): displace for convolutions.
        boudnary ('P'|'O', default='P'): boundary type,
            * 'P', periodic boundary condiction.
            * 'O', open boundary condition.
        is_unitary (bool, default=False): keep unitary if True,\
                here, unitary is defined in the map `U: img_in -> feature_out`.
        var_mask (tuple<bool>, len=2, default=(True,True)):\
                variable mask for weight and bias.

    Attributes:
        weight (ndarray): dimensions are aranged as (feature_out,\
                feature_in, kernel_x, ...), in 'F' order.
        bias (1darray): length of num_feature_out.
        strides (tuple): displace for convolutions.
        boudnary ('P'|'O'): boundary type,
            * 'P', periodic boundary condiction.
            * 'O', open boundary condition.
        is_unitary (bool): keep unitary if True, here, unitary is defined\
                in the map `U: img_in -> feature_out`.
        var_mask (tuple<bool>, len=2): variable mask for weight and bias.

        (Derived):
        csc_indptr (1darray): column pointers for convolution matrix.
        csc_indices (1darray): row indicator for input array.
        weight_indices (1darray): row indicator for filter array\
                (if not contiguous).
    '''
    __display_attrs__ = ['strides', 'boundary',
                         'kernel_shape', 'is_unitary', 'var_mask']

    def __init__(self, input_shape, itype, weight, bias,
                 strides=None, boundary="P",
                 w_contiguous=True, var_mask=(1, 1),
                 is_unitary=False, **kwargs):
        if isinstance(weight, tuple):
            weight = 0.1 * typed_randn(kwargs.get('dtype', itype), weight)
        super(SPConv, self).__init__(input_shape, itype=itype,
                                     weight=weight, bias=bias,
                                     var_mask=var_mask)

        img_nd = self.weight.ndim - 2
        if strides is None:
            strides = (1,) * img_nd
        self.strides = tuple(strides)
        self.boundary = boundary
        self.w_contiguous = w_contiguous
        self.is_unitary = is_unitary

        kernel_shape = self.weight.shape[2:]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(
            kernel_shape, input_shape[-img_nd:], strides, boundary)
        self.output_shape = input_shape[:-img_nd - 1] + \
            (self.num_feature_out,) + self.img_out_shape

        # use the correct fortran subroutine.
        dtype_token = dtype2token(
            np.find_common_type((self.itype, self.dtype), ()))

        if not w_contiguous:
            self.weight_indices = np.asarray(np.tile(np.arange(tuple_prod(
                kernel_shape), dtype='int32'), tuple_prod(img_out_shape)),
                order='F') + 1  # pointer to filter data
            func_f = eval('fspconv.forward_general%s' % dtype_token)
            func_b = eval('fspconv.backward_general%s' % dtype_token)
            func1_f = eval('fspconv.forward1_general%s' % dtype_token)
            func1_b = eval('fspconv.backward1_general%s' % dtype_token)
            self._fforward = lambda *args, **kwargs: func_f(
                *args, weight_indices=self.weight_indices, **kwargs)
            self._fbackward = lambda *args, **kwargs: func_b(
                *args, weight_indices=self.weight_indices, **kwargs)
            self._fforward1 = lambda *args, **kwargs: func1_f(
                *args, weight_indices=self.weight_indices, **kwargs)
            self._fbackward1 = lambda *args, **kwargs: func1_b(
                *args, weight_indices=self.weight_indices, **kwargs)
        else:
            self._fforward = eval('fspconv.forward_contiguous%s' % dtype_token)
            self._fforward1 = eval(
                'fspconv.forward1_contiguous%s' % dtype_token)
            self._fbackward = eval(
                'fspconv.backward_contiguous%s' % dtype_token)
            self._fbackward1 = eval(
                'fspconv.backward1_contiguous%s' % dtype_token)

        # make it unitary
        self.is_unitary = is_unitary
        if is_unitary:
            self.be_unitary()
            self.check_unitary()

    @property
    def img_nd(self):
        '''Dimension of input image.'''
        return len(self.strides)

    @property
    def num_feature_in(self):
        '''Dimension of input feature.'''
        return self.weight.shape[1]

    @property
    def num_feature_out(self):
        '''Dimension of input feature.'''
        return self.weight.shape[0]

    @property
    def kernel_shape(self):
        return self.weight.shape[2:]

    def be_unitary(self):
        weight = self.weight.reshape(self.weight.shape[:2] + (-1,), order='F')
        self.weight = np.asarray(
            np.transpose([np.linalg.qr(weight[:, i].T)[0].T
                          for i in range(
                self.num_feature_in)],
                axes=(1, 0, 2)), order='F')
        self.is_unitary = True

    def check_unitary(self, tol=1e-6):
        # check weight shape
        if self.weight.shape[2] < self.weight.shape[0]:
            raise ValueError('output shape greater than input shape error!')

        # get unitary error
        err = 0
        for i in range(self.num_feature_in):
            weight = self.weight[:, i]
            err += abs(weight.dot(weight.T.conj()) -
                       np.eye(weight.shape[0])).mean()
        err/=self.num_feature_in
        if self.is_unitary and err > tol:
            raise ValueError('non-unitary matrix error, error = %s!' % err)
        return err

    def set_variables(self, variables):
        nw = self.weight.size if self.var_mask[0] else 0
        var1, var2 = variables[:nw], variables[nw:]
        weight_data = self.weight.data if sps.issparse(
            self.weight) else self.weight.ravel(order='F')
        if self.is_unitary and self.var_mask[0]:
            W = self.weight.reshape(self.weight.shape[:2] + (-1,), order='F')
            dG = var1.reshape(W.shape, order='F') - W
            dA = np.einsum('ijk,kjl->ijl', W.T.conj(), dG)
            dA = dA - dA.T.conj()

            B = np.eye(dG.shape[2])[:, None] - dA / 2
            Binv = np.transpose(np.linalg.inv(
                np.transpose(B, axes=(1, 0, 2))), axes=(1, 0, 2))
            Y = np.einsum('ijk,kjl->ijl', W, B.T.conj())
            Y = np.einsum('ijk,kjl->ijl', Y, Binv)

            self.weight[...] = Y.reshape(self.weight.shape, order='F')
        elif self.var_mask[0]:
            weight_data[:] = var1
        if self.var_mask[1]:
            self.bias[:] = var2

    def forward(self, x, **kwargs):
        '''
        Args:
            x (ndarray): (num_batch, nfi, img_in_dims), input in 'F' order.
        Returns:
            ndarray, (num_batch, nfo, img_out_dims), output in 'F' order.
        '''
        x_nd, img_nd = x.ndim, self.img_nd

        # flatten inputs/outputs
        x = x.reshape(x.shape[:x_nd - img_nd] + (-1,), order='F')
        _fltr_flatten = self.weight.reshape(
            self.weight.shape[:2] + (-1,), order='F')

        if x_nd == img_nd + 1:  # single batch wise
            y = self._fforward1(x, csc_indptr=self.csc_indptr,
                                csc_indices=self.csc_indices,
                                fltr_data=_fltr_flatten,
                                bias=self.bias,
                                max_nnz_row=_fltr_flatten.shape[-1])
        else:
            y = self._fforward(x, csc_indptr=self.csc_indptr,
                               csc_indices=self.csc_indices,
                               fltr_data=_fltr_flatten,
                               bias=self.bias,
                               max_nnz_row=_fltr_flatten.shape[-1])
        y = y.reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        '''
        Args:
            xy ((ndarray, ndarray)):
                * x -> (num_batch, nfi, img_in_dims), input in 'F' order.
                * y -> (num_batch, nfo, img_out_dims), output in 'F' order.
            dy (ndarray): (num_batch, nfo, img_out_dims),\
                    gradient of output in 'F' order.
            mask (booleans): (do_xgrad, do_wgrad, do_bgrad).

        Returns:
            tuple(1darray, ndarray): dw, dx
        '''
        x, y = xy
        x_nd, img_nd = x.ndim, self.img_nd
        xpre = x.shape[:x_nd - img_nd]
        ypre = xpre[:-1] + (self.num_feature_out,)
        do_xgrad = True
        mask = self.var_mask

        # flatten inputs/outputs
        x = x.reshape(xpre + (-1,), order='F')
        dy = dy.reshape(ypre + (-1,), order='F')
        _fltr_flatten = self.weight.reshape(
            self.weight.shape[:2] + (-1,), order='F')

        if x_nd == img_nd + 1:  # single batch wise
            dx, dweight, dbias =\
                self._fbackward1(dy, x, self.csc_indptr,
                                 self.csc_indices,
                                 fltr_data=_fltr_flatten,
                                 do_xgrad=do_xgrad,
                                 do_wgrad=mask[0],
                                 do_bgrad=mask[1],
                                 max_nnz_row=_fltr_flatten.shape[-1])
        else:
            dx, dweight, dbias =\
                self._fbackward(dy,
                                x, self.csc_indptr, self.csc_indices,
                                fltr_data=_fltr_flatten,
                                do_xgrad=do_xgrad,
                                do_wgrad=mask[0],
                                do_bgrad=mask[1],
                                max_nnz_row=_fltr_flatten.shape[-1])
        return masked_concatenate([dweight.ravel(order='F'), dbias], mask),\
            dx.reshape(self.input_shape, order='F')


class SPSP(SPConv):
    '''
    Attributes:
        input_shape ((batch, feature_in, img_x, img_y, ...),\
                or (feature_in, img_x, img_y): ...)
        cscmat (csc_matrix): with row indices (feature_in, img_x, img_y, ...),\
                and column indices (feature_out, img_x', img_y', ...)
        bias (1darray): (feature_out), in fortran order.
        strides (tuple): displace for convolutions.

    Attributes (Derived):
        csc_indptr (1darray): column pointers for convolution matrix.
        csc_indices (1darray): row indicator for input array.
        weight_indices (1darray): row indicator for filter array\
                (if not contiguous).
    '''

    def __init__(self, input_shape, itype, cscmat, bias, strides=None,
                 var_mask=(1, 1)):
        self.cscmat = cscmat
        self.bias = bias
        self.var_mask = var_mask

        self.strides = tuple(strides)
        img_nd = len(self.strides)
        if strides is None:
            strides = (1,) * img_nd
        self.boundary = 'P'

        if tuple_prod(input_shape[1:]) != cscmat.shape[0]:
            raise ValueError('csc matrix input shape mismatch!\
                    %s get, but %s desired.' % (
                cscmat.shape[1], tuple_prod(input_shape[1:])))

        # self.csc_indptr, self.csc_indices,
        # self.csc_data = scan2csc_sp(input_shape[1:], strides)
        img_in_shape = input_shape[2:]
        self.csc_indptr, self.csc_indices, self.img_out_shape = spscan2csc(
            kernel_shape, input_shape[-img_nd:], strides, boundary)
        self.img_out_shape = tuple(
            [img_is // stride for img_is, stride in zip(
                img_in_shape, strides)])
        output_shape = input_shape[:1] + \
            (self.num_feature_out,) + self.img_out_shape
        super(SPSP, self).__init__(input_shape, output_shape, itype=itype)

        if self.num_feature_out * tuple_prod(self.img_out_shape
                                             ) != cscmat.shape[1]:
            raise ValueError('csc matrix output shape mismatch! \
%s get, but %s desired.' % (
                cscmat.shape[1], self.num_feature_out * i,
                tuple_prod(self.img_out_shape)))

        # use the correct fortran subroutine.
        dtype_token = dtype2token(
            np.find_common_type((self.itype, self.dtype), ()))

        # select function
        self._fforward = eval('fspsp.forward_conv%s' % dtype_token)
        self._fbackward = eval('fspsp.backward_conv%s' % dtype_token)

    @property
    def img_nd(self):
        '''Dimension of input image.'''
        return len(self.strides)

    @property
    def num_feature_in(self):
        '''Dimension of input feature.'''
        return self.input.shape[1]

    @property
    def num_feature_out(self):
        '''Dimension of input feature.'''
        return self.bias.shape[0]

    def forward(self, x, **kwargs):
        x = x.reshape(xpre + (-1,), order='F')
        y = self._fforward(x, csc_indptr=self.csc_indptr,
                           csc_indices=self.csc_indices,
                           fltr_data=_fltr_flatten,
                           bias=self.bias, max_nnz_row=_fltr_flatten.shape[-1])
        y = y.reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        x = x.reshape(xpre + (-1,), order='F')
        dy = dy.reshape(ypre + (-1,), order='F')
        mask = self.var_mask

        dx, dweight, dbias =\
            self._fbackward(dy, x,
                            self.csc_indptr, self.csc_indices,
                            fltr_data=_fltr_flatten,
                            do_xgrad=True,
                            do_wgrad=mask[0],
                            do_bgrad=mask[1],
                            max_nnz_row=_fltr_flatten.shape[-1])
        return masked_concatenate([dweight.ravel(order='F'),
                                   dbias], mask),\
            dx.reshape(self.input_shape, order='F')


class SPConvProd(LinearBase):
    '''
    Convolutional product layer, the version with variables.
    '''
    __display_attrs__ = ['strides', 'boundary', 'kernel_shape', 'var_mask']

    def __init__(self, input_shape, dtype, weight,
                 bias, strides=None, boundary='O', var_mask=(1, 1), **kwargs):
        super(SPConvProd, self).__init__(input_shape, dtype=dtype,
                                         weight=weight, bias=bias,
                                         var_mask=var_mask)
        self.boundary = boundary

        img_nd = self.weight.ndim - 2
        if strides is None:
            strides = (1,) * img_nd
        self.strides = strides

        kernel_shape = self.weight.shape[2:]
        img_in_shape = input_shape[-img_nd:]
        self.csc_indptr, self.csc_indices, self.img_out_shape = scan2csc(
            kernel_shape, img_in_shape, strides=strides, boundary=boundary)
        output_shape = input_shape[:-img_nd] + self.img_out_shape

        # use the correct fortran subroutine.
        dtype_token = dtype2token(
            np.find_common_type((self.itype, self.dtype), ()))

        # use the correct function
        self._fforward = eval('fspconvprod.forward_%s' % dtype_token)
        self._fbackward = eval('fspconvprod.backward_%s' % dtype_token)

    @property
    def img_nd(self):
        return len(self.strides)

    def forward(self, x, **kwargs):
        '''
        Args:
            x (ndarray): (num_batch, nfi, img_in_dims), input in 'F' order.

        Returns:
            ndarray, (num_batch, nfo, img_out_dims), output in 'F' order.
        '''
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim = tuple_prod(self.input_shape[-img_nd:])
        y = self._fforward(x.reshape([-1, img_dim], order='F'),
                           csc_indptr=self.csc_indptr,
                           weight=self.weight,
                           csc_indices=self.csc_indices
                           ).reshape(self.output_shape, order='F')
        return y

    def backward(self, xy, dy, **kwargs):
        '''It will shed a mask on dy'''
        x, y = xy
        x_nd, img_nd = x.ndim, self.img_nd
        img_dim_in = tuple_prod(self.input_shape[-img_nd:])
        img_dim_out = tuple_prod(self.output_shape[-img_nd:])

        dx = self._fbackward(x=x.reshape([-1, img_dim_in], order='F'),
                             dy=dy.reshape([-1, img_dim_out], order='F'),
                             y=y.reshape([-1, img_dim_out], order='F'),
                             weight=self.weight, csc_indptr=self.csc_indptr,
                             csc_indices=self.csc_indices
                             ).reshape(self.input_shape, order='F')
        return EMPTY_VAR, dx
