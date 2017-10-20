'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.linalg import norm, svd
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose
import pdb
import time

from ..linears import Linear
from ..spconv import SPConv
from ..checks import check_numdiff
from ..utils import typed_randn
from .. import functions, nets, monitors

random.seed(2)


def test_unitary():
    num_batch = 1
    dim_in = 40
    dim_out = 10
    dtype = 'complex128'
    xin_np = asfortranarray(typed_randn(dtype, [num_batch, dim_in]))
    # sv=Linear((num_batch, dim_in),dtype,weight=typed_randn(
    # dtype,(dim_out,dim_in)),bias = typed_randn(dtype, (dim_out,)))
    sv = Linear((num_batch, dim_in), dtype, weight=(
        dim_out, dim_in), bias=0, is_unitary=True)
    print("Testing numdiff for %s" % sv)
    net = nets.ANN(layers=[sv])
    # net.add_layer(monitors.Print)
    net.add_layer(functions.Sum, axis=1)
    net.add_layer(functions.Abs2)

    num_step = 100
    lr = 1e-4
    for i in xrange(num_step):
        data_cache = {}
        y = net.forward(xin_np, data_cache=data_cache)
        dw, dx = net.backward((xin_np, y), data_cache=data_cache)
        net.set_variables(net.get_variables() - dw.conj() * lr)
        if i % 10 == 0:
            err = net.layers[0].check_unitary()
            print('%d, loss = %.4f, unitary err = %s' % (i + 1, y, err))
    assert_almost_equal(y, 0)
    assert_almost_equal(err, 0)


def test_unitary_conv():
    random.seed(2)
    num_batch = 1
    nfi = 1
    dim_out = 10
    img_size = 30
    dtype = 'complex128'
    xin_np = asfortranarray(typed_randn(dtype, [num_batch, nfi, img_size]))
    sv = SPConv((num_batch, nfi, img_size), dtype, weight=typed_randn(
        dtype, (dim_out, nfi, img_size)), bias=0, is_unitary=True)
    print("Testing numdiff for %s" % sv)
    net = nets.ANN(layers=[sv])
    # net.add_layer(monitors.Print)
    net.add_layer(functions.Sum, axis=1)
    net.add_layer(functions.Sum, axis=1)
    net.add_layer(functions.Abs2)

    num_step = 100
    lr = 2e-6
    for i in xrange(num_step):
        data_cache = {}
        y = net.forward(xin_np, data_cache=data_cache)
        dw, dx = net.backward((xin_np, y), data_cache=data_cache)
        net.set_variables(net.get_variables() - dw.conj() * lr)
        if i % 10 == 0:
            err = net.layers[0].check_unitary()
            print('%d, loss = %.4f, unitary err = %s' % (i + 1, y, err))
    assert_almost_equal(y, 0)
    assert_almost_equal(err, 0)


def test_all():
    test_unitary_conv()
    test_unitary()


if __name__ == '__main__':
    test_all()
