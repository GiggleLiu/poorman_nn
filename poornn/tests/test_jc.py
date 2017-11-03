'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose
from scipy import sparse as sps
import pdb
import time

from ..checks import check_numdiff
from ..utils import typed_randn
from ..nets import JointComplex, KeepSignFunc, ANN
from ..linears import Linear
from .. import functions, pfunctions
from ..derivatives import *

random.seed(2)

def test_jcsigmoid():
    itype = 'float32'
    nfi = 8
    nfo = 8
    input_shape = (6, nfi)
    func1 = functions.Sigmoid(input_shape, itype)
    func2 = functions.Reshape(input_shape, itype=itype,
                              output_shape=input_shape)
    func3 = Linear(input_shape, itype=itype, weight=typed_randn(
        itype, (nfo, nfi)), bias=typed_randn(itype, (nfo,)))
    func3_ = Linear(input_shape, itype=itype, weight=typed_randn(
        itype, (nfo, nfi)), bias=typed_randn(itype, (nfo,)))
    jc1 = JointComplex(func1, func1)  # same function
    jc2 = JointComplex(func1, func2)  # different function
    jc3 = JointComplex(func3, func3_)  # functions with parameters

    for jc in [jc1, jc2, jc3]:
        print("Testing numdiff for \n%s" % jc)
        assert_(all(check_numdiff(jc, num_check=100, tol=1e-2)))


def test_keepsign():
    random.seed(4)
    itype = 'float64'
    nfi = 8
    nfo = 8
    input_shape = (6, nfi)
    func1 = functions.Tanh(input_shape, itype)
    func2 = Linear(input_shape, itype=itype, weight=typed_randn(
        itype, (nfo, nfi)), bias=typed_randn(itype, (nfo,)))
    func3 = ANN(layers=[functions.Abs(
        input_shape=input_shape, itype='complex128')])
    func3.add_layer(functions.Tanh)
    jc1 = KeepSignFunc(func1)
    jc2 = KeepSignFunc(func2)

    for jc in [func3, jc1, jc2]:
        print("Testing numdiff for \n%s" % jc)
        assert_(all(check_numdiff(jc, num_check=30, tol=1e-3, eta_w=0.001)))


def test_derivatives():
    funcs = ['KS_Tanh', 'KS_Georgiou1992',
             'JC_Tanh', 'JC_Sigmoid', 'JC_Georgiou1992']
    input_shape = (8, 8)
    itype = 'complex128'
    for func_i in funcs:
        kwargs = {}
        if func_i == 'KS_Georgiou1992':
            kwargs['cr'] = array([0.5, 2.])
        if func_i == 'JC_Georgiou1992':
            kwargs['params'] = array([0.5, 2.])
        func = eval(func_i)(input_shape, itype, **kwargs)
        if func_i == 'KS_Georgiou1992':
            func2 = pfunctions.Georgiou1992(
                input_shape, itype, params=array([0.5, 2]))
            x = array([1, 2, 3.])
            assert_allclose(func.forward(x), func2.forward(x))
        print("Testing numdiff for \n%s" % func)
        assert_(all(check_numdiff(func, num_check=30, tol=1e-2)))


if __name__ == '__main__':
    test_derivatives()
    test_jcsigmoid()
    test_keepsign()
