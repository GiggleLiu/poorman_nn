from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time

from functions import *

def test_sigmoid():
    func=Sigmoid()
    xs=[-1e100,-1e20,-0.5j*pi,-log(2),0,log(2),0.5j*pi,1e20,1e100]
    ys=[0.,0.,1./(1+1j),1./3,0.5,2./3,1./(1-1j),1.,1.]
    dydx=[0.,0.,1j/(1+1j)**2,2./9,0.25,2./9,-1j/(1-1j)**2,0.,0.]
    assert_allclose(func.forward(xs),ys)
    assert_allclose(func.backward(xs,ys)[1],dydx)
    assert_allclose([func(x) for x in xs],ys)

def test_log2cosh():
    func=Log2cosh()
    xs=[-1e100,-1e20,-0.25j*pi,-log(2),0,log(2),0.25j*pi,1e20,1e100]
    ys=[1e100,1e20,log(2.)/2,log(2.5),log(2.),log(2.5),log(2.)/2,1e20,1e100]
    dydx=tanh(xs)
    assert_allclose(func.forward(xs),ys)
    assert_allclose(func.backward(xs,ys)[1],dydx)
    assert_allclose([func(x) for x in xs],ys)

def test_all():
    test_sigmoid()
    test_log2cosh()

test_all()
